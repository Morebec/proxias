package agentdemo

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"iter"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters"
	agentanthropic "github.com/morebec/proxias/adapters/anthropic"
)

// The strategies an example runs on (-strategy).
const (
	// StrategyReAct is the ReAct strategy: one model call a turn, every action
	// a tool with its JSON Schema.
	StrategyReAct = "react"
	// StrategyAdaptive is the adaptive strategy, its selector Claude in one
	// call (proxias.LLMStrategySelector), its candidates ReAct, ReWOO and
	// Observe-Plan-Execute.
	StrategyAdaptive = "adaptive"
	// StrategyAdaptiveSplit is the adaptive strategy, its selector Claude
	// answering as a System One model (proxias.System1StrategySelector of
	// adapters.LLMSystem1).
	StrategyAdaptiveSplit = "adaptive-split"
	// StrategyReWOO is the ReWOO strategy: a plan of the whole task, its steps
	// run, then the answer (proxias.ReWOOCognitiveStrategy).
	StrategyReWOO = "rewoo"
	// StrategyObservePlanExecute is the Observe-Plan-Execute strategy: a plan
	// revised after every step, then the answer
	// (proxias.ObservePlanExecuteCognitiveStrategy).
	StrategyObservePlanExecute = "observe-plan-execute"
)

// Strategies are all the strategies, in the order of the bench.
var Strategies = []string{
	StrategyReAct,
	StrategyAdaptive,
	StrategyAdaptiveSplit,
	StrategyReWOO,
	StrategyObservePlanExecute,
}

// aliases are the names the strategies had as shells.
var aliases = map[string]string{
	"react-tools": StrategyReAct,
}

var strategyKind = flag.String("strategy", StrategyReAct, "the strategy of the agents: "+strings.Join(Strategies, ", "))

// Env is what the strategies of a run share: the counter of their calls to
// Claude, and how often a call is retried.
// Separate runs have separate envs, so that nothing is shared between them.
type Env struct {
	Counter *agentanthropic.UsageCounter
	// MaxRetries is the retries of a failed call (the SDK's own when negative):
	// 0 so that a rate limit is an error.
	MaxRetries int
}

// Counter counts the usage of every call to Claude of the examples.
var Counter agentanthropic.UsageCounter

// Script is what a scripted agent answers, on each strategy: the ReAct model's
// answers (React, written in the markers of parseAnswer), or, on the
// strategies that plan, the requests of each task (Requests), what the answer
// says once the actions ran (Syntheses), and what it says to a task that needs
// no action (Reasoning).
type Script struct {
	React     []string
	Requests  [][]proxias.ActionExecutionRequest
	Syntheses []string
	Reasoning []string
}

// Strategy is the strategy of an agent, of the kind of -strategy, on Claude
// with -claude, otherwise on the script.
func Strategy(script Script) proxias.CognitiveStrategy {
	parse()
	return StrategyOf(*strategyKind, *claude, script)
}

// StrategyOf is the strategy of an agent of the kind, on Claude or on the
// script, counted by Counter.
func StrategyOf(kind string, onClaude bool, script Script) proxias.CognitiveStrategy {
	return Env{
		Counter:    &Counter,
		MaxRetries: -1,
	}.StrategyOf(
		kind,
		onClaude,
		script,
	)
}

// StrategyOf is the strategy of an agent of the kind, on Claude or on the
// script, in the env.
func (env Env) StrategyOf(kind string, onClaude bool, script Script) proxias.CognitiveStrategy {
	parse()
	if alias, ok := aliases[kind]; ok {
		kind = alias
	}
	switch kind {
	case StrategyReAct:
		return proxias.NewReActStrategy(env.llm(onClaude, script.React))
	case StrategyAdaptive:
		if onClaude {
			claude := env.llm(true, nil)
			return proxias.NewAdaptiveStrategy(
				proxias.LLMStrategySelector{LLM: claude},
				adaptiveCandidates(
					proxias.NewReActStrategy(claude),
					proxias.NewReWOOStrategy(claude),
					proxias.NewObservePlanExecuteStrategy(claude),
				)...,
			)
		}
		// On the script, Claude chooses ReWOO for every task: the scripts of the
		// candidates are each in the order of the tasks, so one candidate runs
		// them all.
		choices := make([]string, 0, 64)
		for range 64 {
			choices = append(choices, `{"strategy":"rewoo"}`+Done)
		}
		return proxias.NewAdaptiveStrategy(
			proxias.LLMStrategySelector{LLM: scriptedLLM(choices)},
			adaptiveCandidates(
				proxias.NewReActStrategy(scriptedLLM(script.React)),
				proxias.NewReWOOStrategy(reWOOScript(script)),
				proxias.NewObservePlanExecuteStrategy(observePlanExecuteScript(script)),
			)...,
		)
	case StrategyAdaptiveSplit:
		if onClaude {
			claude := env.llm(true, nil)
			return proxias.NewAdaptiveStrategy(
				proxias.System1StrategySelector{
					System1: adapters.LLMSystem1{LLM: claude},
				},
				adaptiveCandidates(
					proxias.NewReActStrategy(claude),
					proxias.NewReWOOStrategy(claude),
					proxias.NewObservePlanExecuteStrategy(claude),
				)...,
			)
		}
		// On the script, System1 chooses ReWOO for every task, sure of it.
		choices := make([]string, 0, 64)
		for range 64 {
			choices = append(choices, `{"choice":"rewoo","confidence":1}`+Done)
		}
		return proxias.NewAdaptiveStrategy(
			proxias.System1StrategySelector{
				System1: adapters.LLMSystem1{LLM: scriptedLLM(choices)},
			},
			adaptiveCandidates(
				proxias.NewReActStrategy(scriptedLLM(script.React)),
				proxias.NewReWOOStrategy(reWOOScript(script)),
				proxias.NewObservePlanExecuteStrategy(observePlanExecuteScript(script)),
			)...,
		)
	case StrategyReWOO:
		if onClaude {
			return proxias.NewReWOOStrategy(env.llm(true, nil))
		}
		return proxias.NewReWOOStrategy(reWOOScript(script))
	case StrategyObservePlanExecute:
		if onClaude {
			return proxias.NewObservePlanExecuteStrategy(env.llm(true, nil))
		}
		return proxias.NewObservePlanExecuteStrategy(observePlanExecuteScript(script))
	default:
		log.Fatalf("-strategy %q: it is one of %s", kind, strings.Join(Strategies, ", "))
		return nil
	}
}

// adaptiveCandidates are the candidates of the adaptive strategy, the first
// one its default.
func adaptiveCandidates(react, reWOO, observePlanExecute proxias.CognitiveStrategy) []proxias.StrategyCandidate {
	return []proxias.StrategyCandidate{
		{
			Name: StrategyReAct,
			Description: "Decides each step after seeing the last one's result, one model call a step: " +
				"conversations, questions answered from the conversation, open-ended tasks.",
			Strategy: react,
		},
		{
			Name: StrategyReWOO,
			Description: "Plans every step up front, runs them, then answers, in two model calls: " +
				"tasks of one or more actions whose steps are known from the start.",
			Strategy: reWOO,
		},
		{
			Name: StrategyObservePlanExecute,
			Description: "Keeps a written plan and revises it after every step: " +
				"longer tasks that must adapt to what each step finds.",
			Strategy: observePlanExecute,
		},
	}
}

// reWOOScript is the scripted Claude of the ReWOO strategy: its plans are the
// requests of each turn it plans, then none; its answers are a turn's
// synthesis when it ran actions and its reasoning otherwise, then the
// reasoning left (an agent without actions is asked no plan).
func reWOOScript(script Script) proxias.LLM {
	var plans, answers []string
	synthesis, reasoning := 0, 0
	for _, requests := range script.Requests {
		steps := make([]map[string]any, 0, len(requests))
		for i, request := range requests {
			step := map[string]any{
				"id":     fmt.Sprintf("E%d", i+1),
				"action": request.ActionID,
				"input":  request.Params,
			}
			steps = append(steps, step)
		}
		plan, _ := json.Marshal(map[string]any{"steps": steps}) // the params of a script marshal
		plans = append(plans, string(plan)+Done)

		if len(requests) > 0 && synthesis < len(script.Syntheses) {
			answers = append(answers, script.Syntheses[synthesis]+Done)
			synthesis++
			continue
		}
		if reasoning < len(script.Reasoning) {
			answers = append(answers, script.Reasoning[reasoning])
			reasoning++
		}
	}
	for range 32 {
		plans = append(plans, `{"steps":[]}`+Done)
	}
	answers = append(answers, script.Reasoning[reasoning:]...)

	return plannedLLM{
		planner:  scriptedLLM(plans),
		answerer: scriptedLLM(answers),
	}
}

// observePlanExecuteScript is the scripted Claude of the Observe-Plan-Execute
// strategy: its plans are, for each turn it plans, the requests left after
// each step, then none; its answers are the task's synthesis when it ran
// actions and its reasoning otherwise, then the reasoning left (an agent
// without actions is asked no plan).
func observePlanExecuteScript(script Script) proxias.LLM {
	var plans, answers []string
	synthesis, reasoning := 0, 0
	for _, requests := range script.Requests {
		for left := range requests {
			steps := make([]map[string]any, 0, len(requests)-left)
			for _, request := range requests[left:] {
				step := map[string]any{
					"action": request.ActionID,
					"input":  request.Params,
				}
				steps = append(steps, step)
			}
			plan, _ := json.Marshal(map[string]any{"steps": steps}) // the params of a script marshal
			plans = append(plans, string(plan)+Done)
		}
		plans = append(plans, `{"steps":[]}`+Done)

		if len(requests) > 0 && synthesis < len(script.Syntheses) {
			answers = append(answers, script.Syntheses[synthesis]+Done)
			synthesis++
			continue
		}
		if reasoning < len(script.Reasoning) {
			answers = append(answers, script.Reasoning[reasoning])
			reasoning++
		}
	}
	for range 32 {
		plans = append(plans, `{"steps":[]}`+Done)
	}
	answers = append(answers, script.Reasoning[reasoning:]...)

	return plannedLLM{
		planner:  scriptedLLM(plans),
		answerer: scriptedLLM(answers),
	}
}

// plannedLLM is the scripted model of a strategy that plans: the planner
// answers the calls whose system prompt opens with a plan prompt, the answerer
// the others.
type plannedLLM struct {
	planner  proxias.LLM
	answerer proxias.LLM
}

// Complete asks the planner or the answerer, as the call's system prompt says.
func (l plannedLLM) Complete(ctx context.Context, req proxias.LLMRequest) iter.Seq2[proxias.LLMChunk, error] {
	if isPlanRequest(req) {
		return l.planner.Complete(ctx, req)
	}
	return l.answerer.Complete(ctx, req)
}

// isPlanRequest tells whether the request opens with a plan prompt.
func isPlanRequest(req proxias.LLMRequest) bool {
	if len(req.Messages) == 0 {
		return false
	}
	for _, part := range req.Messages[0].Parts {
		if strings.HasPrefix(part.Text, proxias.DefaultReWOOPlanPrompt) {
			return true
		}
		if strings.HasPrefix(part.Text, proxias.DefaultObservePlanExecutePlanPrompt) {
			return true
		}
	}
	return false
}

// Client is the client of Claude of the examples, which Counter counts: its key
// is ANTHROPIC_API_KEY, set to devKey
// when it is not.
func Client() anthropicsdk.Client {
	return Env{Counter: &Counter, MaxRetries: -1}.Client()
}

// Client is the client of Claude of the env, which its counter counts: its key
// is ANTHROPIC_API_KEY, set to devKey
// when it is not.
func (env Env) Client() anthropicsdk.Client {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		_ = os.Setenv("ANTHROPIC_API_KEY", devKey)
	}
	options := []option.RequestOption{
		option.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
	}
	if env.Counter != nil {
		options = append(options, option.WithMiddleware(env.Counter.Middleware()))
	}
	if env.MaxRetries >= 0 {
		options = append(options, option.WithMaxRetries(env.MaxRetries))
	}
	return anthropicsdk.NewClient(options...)
}

// llm is Claude, or a model that answers the script.
func (env Env) llm(onClaude bool, script []string) proxias.LLM {
	if onClaude {
		return agentanthropic.NewLLM(env.Client(), ClaudeModel)
	}
	return scriptedLLM(script)
}

// scriptedLLM answers each call with the next of the answers, written in the
// markers of parseAnswer: thoughts and a final answer as text, actions as tool
// calls; then with nothing. It is Claude, served the answers by a local server.
func scriptedLLM(answers []string) proxias.LLM {
	var (
		mutex sync.Mutex
		next  int
	)
	server := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			mutex.Lock()
			answer := Done
			if next < len(answers) {
				answer = answers[next]
				next++
			}
			call := next
			mutex.Unlock()
			content := []map[string]any{}
			stopReason := "end_turn"
			for i, e := range parseAnswer(answer) {
				if e.Type != proxias.EventTypeActionRequest {
					if e.Text() != "" {
						content = append(
							content,
							map[string]any{"type": "text", "text": e.Text()},
						)
					}
					continue
				}
				var request proxias.ActionExecutionRequest
				if err := json.Unmarshal(e.Payload, &request); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				stopReason = "tool_use" // as Claude says when it calls tools
				input := request.Params
				if input == nil {
					input = map[string]any{}
				}
				content = append(
					content,
					map[string]any{
						"type":  "tool_use",
						"id":    fmt.Sprintf("toolu_%d_%d", call, i),
						"name":  request.ActionID,
						"input": input,
					},
				)
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":          "msg",
				"type":        "message",
				"role":        "assistant",
				"model":       ClaudeModel,
				"stop_reason": stopReason,
				"usage":       map[string]any{}, "content": content,
			})
		}),
	)
	return agentanthropic.NewLLM(
		anthropicsdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("script"), option.WithMaxRetries(0)),
		ClaudeModel,
	)
}

// Requests are the requests of a turn: one, of the action, its parameters as
// pairs of name and value.
func Requests(actionID string, params ...any) []proxias.ActionExecutionRequest {
	p := map[string]any{}
	for i := 0; i+1 < len(params); i += 2 {
		p[fmt.Sprint(params[i])] = params[i+1]
	}
	return []proxias.ActionExecutionRequest{
		{ActionID: actionID, Params: p},
	}
}

// Actions is an executor of the actions, by their ID: each runs a function of
// its parameters, whose output is the
// data of its result ({"output": ...}), or whose error fails it.
func Actions(actions ...Action) proxias.ActionExecutor {
	executor := proxias.SimpleActionExecutor{
		Actions: map[string]proxias.SimpleAction{},
	}
	for _, action := range actions {
		run := action.Run
		executor.Actions[action.ID] = proxias.SimpleAction{
			ActionDescriptor: action.ActionDescriptor,
			RunFn: func(
				ctx context.Context,
				_ proxias.Session,
				params map[string]any,
			) (proxias.ActionExecutionResult, error) {
				output, err := run(ctx, params)
				if err != nil {
					// A failure the agent is told of, not an error of the
					// executor.
					//nolint:nilerr // its result says why
					return proxias.ActionExecutionResult{
						ActionID: action.ID,
						Status:   "failed",
						Data:     map[string]any{"error": err.Error()},
					}, nil
				}
				return proxias.ActionExecutionResult{
					ActionID: action.ID,
					Status:   "completed",
					Data:     map[string]any{"output": output},
				}, nil
			},
		}
	}
	return executor
}

// Action is an action of Actions.
type Action struct {
	proxias.ActionDescriptor
	Run func(ctx context.Context, params map[string]any) (any, error)
}

// Params is the JSON Schema of an object of string parameters, all required:
// pairs of name and description.
func Params(pairs ...string) map[string]any {
	properties := map[string]any{}
	var required []string
	for i := 0; i+1 < len(pairs); i += 2 {
		properties[pairs[i]] = map[string]any{
			"type":        "string",
			"description": pairs[i+1],
		}
		required = append(required, pairs[i])
	}
	return map[string]any{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}
}

func parse() {
	if !flag.Parsed() {
		flag.Parse()
	}
}
