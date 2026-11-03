package agentdemo

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/agentsdk"
	agentanthropic "github.com/morebec/proxias/agentsdk/anthropic"
)

// The strategies an example runs on (-strategy).
const (
	// StrategyReAct is the ReAct strategy: one model call a turn, every action a tool with its JSON Schema.
	StrategyReAct = "react"
	// StrategyGated is the gated strategy, its gate Claude answering as a System One model (agentsdk.LLMSystem1).
	StrategyGated = "gated"
)

// Strategies are all the strategies, in the order of the bench.
var Strategies = []string{StrategyReAct, StrategyGated}

// aliases are the names the strategies had as shells.
var aliases = map[string]string{"react-tools": StrategyReAct, "proxias": StrategyGated}

var strategyKind = flag.String("strategy", StrategyReAct, "the strategy of the agents: "+strings.Join(Strategies, ", "))

// Env is what the strategies of a run share: the counter of their calls to Claude, the timer of the gated stages, and
// how often a call is retried. Separate runs have separate envs, so that nothing is shared between them.
type Env struct {
	Counter    *agentanthropic.UsageCounter
	StageTimer func(agentsdk.GatedStage, time.Duration)
	// MaxRetries is the retries of a failed call (the SDK's own when negative): 0 so that a rate limit is an error.
	MaxRetries int
}

// Counter counts the usage of every call to Claude of the examples.
var Counter agentanthropic.UsageCounter

// Script is what a scripted agent answers, on each strategy: the ReAct model's answers (React, written in the
// markers of parseAnswer), or, on the gated strategy, what its gate finds (one intent per turn it gates, then none: its
// action is the gate's choice, its parameters what the synthesizer extracts), what its synthesizer says once the
// actions ran, and what its fallback, the ReAct strategy, answers.
type Script struct {
	React     []string
	Intents   []proxias.Intent
	Syntheses []string
	Fallback  []string
}

// Strategy is the strategy of an agent, of the kind of -strategy, on Claude with -claude, otherwise on the script.
func Strategy(script Script) agentsdk.CognitiveStrategy {
	parse()
	return StrategyOf(*strategyKind, *claude, script)
}

// StrategyOf is the strategy of an agent of the kind, on Claude or on the script, counted by Counter.
func StrategyOf(kind string, onClaude bool, script Script) agentsdk.CognitiveStrategy {
	return Env{Counter: &Counter, MaxRetries: -1}.StrategyOf(kind, onClaude, script)
}

// StrategyOf is the strategy of an agent of the kind, on Claude or on the script, in the env.
func (env Env) StrategyOf(kind string, onClaude bool, script Script) agentsdk.CognitiveStrategy {
	parse()
	if alias, ok := aliases[kind]; ok {
		kind = alias
	}
	switch kind {
	case StrategyReAct:
		return agentsdk.NewReActStrategy(env.llm(onClaude, script.React))
	case StrategyGated:
		var opts []agentsdk.GatedOption
		if env.StageTimer != nil {
			opts = append(opts, agentsdk.WithStageTimer(env.StageTimer))
		}
		if onClaude {
			claude := env.llm(true, nil)
			return agentsdk.NewGatedStrategy(agentsdk.LLMSystem1{LLM: claude}, claude, agentsdk.NewReActStrategy(claude), opts...)
		}
		// On the script, the gate answers the questions of each intent, then that nothing is asked for: an answer for
		// both the choice and the noul, twice, as they are asked concurrently; the synthesizer calls the actions of
		// each intent, then says its synthesis.
		var gate, synthesizer []string
		synthesis := 0
		for _, intent := range script.Intents {
			if len(intent.ActionExecutions) == 0 {
				gate = append(gate, `{"choice":"","confidence":0,"noul":0}`+Done, `{"choice":"","confidence":0,"noul":0}`+Done)
				continue
			}
			answer := `{"choice":"` + intent.ActionExecutions[0].ActionID + `","confidence":1,"noul":1}` + Done
			gate = append(gate, answer, answer)
			var calls string
			for _, e := range intent.ActionExecutions {
				request, _ := json.Marshal(e) // the params of a script marshal
				calls += actionOpen + string(request) + actionClose
			}
			synthesizer = append(synthesizer, calls)
			if synthesis < len(script.Syntheses) {
				synthesizer = append(synthesizer, script.Syntheses[synthesis]+Done)
				synthesis++
			}
		}
		for range 32 {
			gate = append(gate, `{"choice":"","confidence":0,"noul":0}`+Done)
		}
		return agentsdk.NewGatedStrategy(agentsdk.LLMSystem1{LLM: scriptedLLM(gate)}, scriptedLLM(synthesizer), agentsdk.NewReActStrategy(scriptedLLM(script.Fallback)), opts...)
	default:
		log.Fatalf("-strategy %q: it is one of %s", kind, strings.Join(Strategies, ", "))
		return nil
	}
}

// Client is the client of Claude of the examples, which Counter counts: its key is ANTHROPIC_API_KEY, set to devKey
// when it is not.
func Client() anthropicsdk.Client {
	return Env{Counter: &Counter, MaxRetries: -1}.Client()
}

// Client is the client of Claude of the env, which its counter counts: its key is ANTHROPIC_API_KEY, set to devKey
// when it is not.
func (env Env) Client() anthropicsdk.Client {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		_ = os.Setenv("ANTHROPIC_API_KEY", devKey)
	}
	options := []option.RequestOption{option.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY"))}
	if env.Counter != nil {
		options = append(options, option.WithMiddleware(env.Counter.Middleware()))
	}
	if env.MaxRetries >= 0 {
		options = append(options, option.WithMaxRetries(env.MaxRetries))
	}
	return anthropicsdk.NewClient(options...)
}

// llm is Claude, or a model that answers the script.
func (env Env) llm(onClaude bool, script []string) agentsdk.LLM {
	if onClaude {
		return agentanthropic.NewLLM(env.Client(), ClaudeModel)
	}
	return scriptedLLM(script)
}

// scriptedLLM answers each call with the next of the answers, written in the markers of parseAnswer: thoughts and a
// final answer as text, actions as tool calls; then with nothing. It is Claude, served the answers by a local server.
func scriptedLLM(answers []string) agentsdk.LLM {
	var (
		mutex sync.Mutex
		next  int
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mutex.Lock()
		answer := Done
		if next < len(answers) {
			answer = answers[next]
			next++
		}
		call := next
		mutex.Unlock()
		content := []map[string]any{}
		for i, e := range parseAnswer(answer) {
			if e.Type != agentsdk.EventTypeActionRequest {
				if e.Text() != "" {
					content = append(content, map[string]any{"type": "text", "text": e.Text()})
				}
				continue
			}
			var request agentsdk.ActionExecutionRequest
			if err := json.Unmarshal(e.Payload, &request); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			input := request.Params
			if input == nil {
				input = map[string]any{}
			}
			content = append(content, map[string]any{"type": "tool_use", "id": fmt.Sprintf("toolu_%d_%d", call, i), "name": request.ActionID, "input": input})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "msg", "type": "message", "role": "assistant", "model": ClaudeModel, "stop_reason": "end_turn",
			"usage": map[string]any{}, "content": content,
		})
	}))
	return agentanthropic.NewLLM(anthropicsdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("script"), option.WithMaxRetries(0)), ClaudeModel)
}

// Intent is an intent of one action, its parameters as pairs of name and value.
func Intent(actionID string, params ...any) proxias.Intent {
	p := map[string]any{}
	for i := 0; i+1 < len(params); i += 2 {
		p[fmt.Sprint(params[i])] = params[i+1]
	}
	return proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{{ActionID: actionID, Params: p}}}
}

// Actions is an executor of the actions, by their ID: each runs a function of its parameters, whose output is the
// data of its result ({"output": ...}), or whose error fails it.
func Actions(actions ...Action) agentsdk.ActionExecutor {
	executor := agentsdk.SimpleActionExecutor{Actions: map[string]agentsdk.SimpleAction{}}
	for _, action := range actions {
		run := action.Run
		executor.Actions[action.ID] = agentsdk.SimpleAction{
			ActionDescriptor: action.ActionDescriptor,
			RunFn: func(ctx context.Context, _ agentsdk.Session, params map[string]any) (agentsdk.ActionExecutionResult, error) {
				output, err := run(ctx, params)
				if err != nil {
					// A failure the agent is told of, not an error of the executor.
					return agentsdk.ActionExecutionResult{ActionID: action.ID, Status: "failed", Data: map[string]any{"error": err.Error()}}, nil //nolint:nilerr // the action failed: its result says why
				}
				return agentsdk.ActionExecutionResult{ActionID: action.ID, Status: "completed", Data: map[string]any{"output": output}}, nil
			},
		}
	}
	return executor
}

// Action is an action of Actions.
type Action struct {
	agentsdk.ActionDescriptor
	Run func(ctx context.Context, params map[string]any) (any, error)
}

// Params is the JSON Schema of an object of string parameters, all required: pairs of name and description.
func Params(pairs ...string) map[string]any {
	properties := map[string]any{}
	var required []string
	for i := 0; i+1 < len(pairs); i += 2 {
		properties[pairs[i]] = map[string]any{"type": "string", "description": pairs[i+1]}
		required = append(required, pairs[i])
	}
	return map[string]any{"type": "object", "properties": properties, "required": required}
}

func parse() {
	if !flag.Parsed() {
		flag.Parse()
	}
}
