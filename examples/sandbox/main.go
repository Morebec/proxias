// Command harness is a sandbox for the middleware chain: it serves a ControlPlane as the OpenAI chat completions
// endpoint, so that a real harness (OpenClaw, see `task sandbox:openclaw`) can talk to it. To try a middleware,
// change buildControlPlane and restart `task sandbox`.
//
// The workflow is say_hello_world. Without ANTHROPIC_API_KEY, the requests that no workflow fits are answered by
// a stub that lists the workflows, which shows what the reasoning engine is given.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
)

func main() {
	controlPlane, err := buildControlPlane()
	if err != nil {
		log.Fatal(err)
	}

	addr := os.Getenv("PROXIAS_ADDR")
	if addr == "" {
		addr = ":8787"
	}
	mux := http.NewServeMux()
	mux.Handle("POST /v1/chat/completions", proxias.NewChatCompletionHandler(controlPlane))
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintln(w, "proxias sandbox: POST /v1/chat/completions is the endpoint a harness talks to.")
	})

	log.Printf("proxias sandbox serving POST /v1/chat/completions on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

// buildControlPlane is the chain to try. Middleware run in order, and the first one that answers a request
// bypasses the rest: put a new middleware in the list where it should see the requests.
func buildControlPlane() (proxias.ControlPlane, error) {
	interception, err := proxias.NewWorkflowInterceptionMiddleware(
		proxias.IntentClassifierFunc(classify),
		workflowEngine{},
		proxias.WorkflowExecutionSynthesizerFunc(synthesize),
	)
	if err != nil {
		return nil, err
	}

	reasoning, err := proxias.NewReasoningMiddleware(reasoner())
	if err != nil {
		return nil, err
	}

	return proxias.NewControlPlane(logRequests(), confirmRestart(), interception, reasoning)
}

// logRequests logs what each request costs and how it ends: never its messages, which are data.
func logRequests() proxias.Middleware {
	return func(next proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			start := time.Now()
			response, err := next(ctx, req)
			log.Printf("request model=%q stream=%t status=%d took=%s err=%v", req.Model, req.Stream, response.Status, time.Since(start).Round(time.Millisecond), err)
			return response, err
		}
	}
}

// classify is a keyword IntentClassifier: a request that says hello asks for the say_hello_world workflow. A harness
// adds messages of its own, such as its context, after the message of the user: so the last two messages of the user
// are checked, not only the last one.
func classify(_ context.Context, req proxias.ChatCompletionRequest, _ []proxias.WorkflowDescriptor) (proxias.Intent, error) {
	for _, text := range recentUserTexts(req) {
		if strings.Contains(strings.ToLower(text), "hello") {
			return proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{{WorkflowID: "say_hello_world"}}}, nil
		}
	}
	return proxias.Intent{}, nil
}

// workflowEngine is the WorkflowExecutionEngine of the sandbox: one workflow.
type workflowEngine struct{}

func (workflowEngine) WorkflowDescriptors(context.Context, proxias.ChatCompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return []proxias.WorkflowDescriptor{{ID: "say_hello_world", Name: "Say hello world", Description: "Says hello to the world"}}, nil
}

func (workflowEngine) ExecuteWorkflows(_ context.Context, _ proxias.ChatCompletionRequest, executions []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
	results := make([]proxias.WorkflowExecutionResult, 0, len(executions))
	for _, execution := range executions {
		results = append(results, proxias.WorkflowExecutionResult{WorkflowID: execution.WorkflowID, Status: "completed", Data: map[string]any{"message": "Hello, World!"}})
	}
	return results, nil
}

// synthesize replies with what the workflow answered, where a real one would make a single LLM call.
func synthesize(_ context.Context, synthesis proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
	if len(synthesis.ExecutionResults) == 0 {
		return proxias.ChatCompletionResponse{}, errors.New("no result to synthesize")
	}
	return reply(synthesis.Request, fmt.Sprint(synthesis.ExecutionResults[0].Data["message"]))
}

// reasoner is the reasoning engine: Claude when ANTHROPIC_API_KEY is set, otherwise a stub that lists the
// workflows it is given.
func reasoner() proxias.ReasoningEngine {
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		return anthropic.NewReasoningEngine(anthropicsdk.NewClient(), anthropicsdk.ModelClaudeHaiku4_5)
	}
	return proxias.ReasoningEngineFunc(func(_ context.Context, req proxias.ChatCompletionRequest, descs []proxias.WorkflowDescriptor) (proxias.ChatCompletionResponse, error) {
		names := make([]string, 0, len(descs))
		for _, d := range descs {
			names = append(names, d.Name)
		}
		return reply(req, "(stub, no LLM) I can: "+strings.Join(names, ", "))
	})
}

// reply answers req with text, as a chat completion, or as a stream of one if the harness asked for a stream.
func reply(req proxias.ChatCompletionRequest, text string) (proxias.ChatCompletionResponse, error) {
	const id = "proxias-sandbox"
	created := time.Now().Unix()
	if !req.Stream {
		return proxias.NewChatCompletionResponse(proxias.ChatCompletion{
			ID: id, Object: "chat.completion", Created: created, Model: req.Model,
			Choices: []proxias.ChatCompletionChoice{{
				Message:      proxias.ChatMessage{Role: proxias.ChatMessageRoleAssistant, Content: proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: text}}},
				FinishReason: proxias.ChatFinishReasonStop,
			}},
		})
	}

	stop := proxias.ChatFinishReasonStop
	return proxias.NewChatCompletionStream(func(emit func(proxias.ChatCompletionChunk) error) error {
		chunk := func(delta proxias.ChatCompletionDelta, finish *proxias.ChatFinishReason) proxias.ChatCompletionChunk {
			return proxias.ChatCompletionChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
				Choices: []proxias.ChatCompletionChunkChoice{{Delta: delta, FinishReason: finish}}}
		}
		if err := emit(chunk(proxias.ChatCompletionDelta{Role: proxias.ChatMessageRoleAssistant}, nil)); err != nil {
			return err
		}
		if err := emit(chunk(proxias.ChatCompletionDelta{Content: text}, nil)); err != nil {
			return err
		}
		return emit(chunk(proxias.ChatCompletionDelta{}, &stop))
	}), nil
}

// recentUserTexts are the texts of the last two messages of the user.
func recentUserTexts(req proxias.ChatCompletionRequest) []string {
	var users []string
	for _, m := range req.Messages {
		if m.Role == proxias.ChatMessageRoleUser {
			users = append(users, m.Content.Text())
		}
	}
	return users[max(0, len(users)-2):]
}

// confirmRestart asks the user before a restart, with the ask_user tool of OpenClaw: a request that asks for a
// restart, from a harness that declares ask_user, is answered with a call of that tool. When the harness sends the
// answer back, as a tool message, the restart is confirmed or dropped in text. Anything else goes on.
func confirmRestart() proxias.Middleware {
	return func(next proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			if n := len(req.Messages); n > 0 && req.Messages[n-1].Role == proxias.ChatMessageRoleTool && strings.HasPrefix(req.Messages[n-1].ToolCallID, confirmRestartCallID) {
				return reply(req, restartAnswer(req.Messages[n-1].Content.Text()))
			}
			if !hasTool(req, "ask_user") || !wantsRestart(req) {
				return next(ctx, req)
			}
			return askUser(req)
		}
	}
}

func hasTool(req proxias.ChatCompletionRequest, name string) bool {
	for _, t := range req.Tools {
		if t.Function.Name == name {
			return true
		}
	}
	return false
}

func wantsRestart(req proxias.ChatCompletionRequest) bool {
	for _, text := range recentUserTexts(req) {
		if strings.Contains(strings.ToLower(text), "restart") {
			return true
		}
	}
	return false
}

// restartAnswer reads the answer that ask_user returned: the first option of the question is the one that restarts.
func restartAnswer(result string) string {
	var answer struct {
		Status  string `json:"status"`
		Answers struct {
			Answers map[string][]string `json:"answers"`
		} `json:"answers"`
	}
	if err := json.Unmarshal([]byte(result), &answer); err != nil || answer.Status != "answered" {
		return "The restart was not confirmed, so nothing was restarted."
	}
	for _, choices := range answer.Answers.Answers {
		for _, choice := range choices {
			if strings.HasPrefix(choice, "Yes") {
				return "api restarted (confirmed by you)."
			}
		}
	}
	return "The restart was declined, so nothing was restarted."
}

// confirmRestartCallID identifies the call of ask_user that confirmRestart makes, so that its answer is recognised.
// A harness may rewrite the IDs of its calls, so the ID has only letters and digits, and its answer is matched by
// prefix: OpenClaw sends back a suffixed copy of it.
const confirmRestartCallID = "proxiasconfirmrestart"

// askUser answers with a call of the ask_user tool, asking the user to confirm the restart.
func askUser(req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
	arguments := `{"questions":[{"id":"confirm_restart","header":"Restart","question":"Restart the api service now?","options":[{"label":"Yes, restart (Recommended)","description":"Restart the api service"},{"label":"No","description":"Leave it running"}]}]}`
	call := proxias.ChatToolCall{ID: confirmRestartCallID, Type: "function", Function: proxias.ChatFunctionCall{Name: "ask_user", Arguments: arguments}}
	return replyWithToolCall(req, call)
}

// replyWithToolCall answers req with a call of a tool, as a chat completion or as a stream of one.
func replyWithToolCall(req proxias.ChatCompletionRequest, call proxias.ChatToolCall) (proxias.ChatCompletionResponse, error) {
	const id = "proxias-sandbox"
	created := time.Now().Unix()
	finish := proxias.ChatFinishReasonToolCalls
	if !req.Stream {
		return proxias.NewChatCompletionResponse(proxias.ChatCompletion{
			ID: id, Object: "chat.completion", Created: created, Model: req.Model,
			Choices: []proxias.ChatCompletionChoice{{
				Message:      proxias.ChatMessage{Role: proxias.ChatMessageRoleAssistant, ToolCalls: []proxias.ChatToolCall{call}},
				FinishReason: finish,
			}},
		})
	}

	return proxias.NewChatCompletionStream(func(emit func(proxias.ChatCompletionChunk) error) error {
		chunk := func(delta proxias.ChatCompletionDelta, reason *proxias.ChatFinishReason) proxias.ChatCompletionChunk {
			return proxias.ChatCompletionChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
				Choices: []proxias.ChatCompletionChunkChoice{{Delta: delta, FinishReason: reason}}}
		}
		var start proxias.ChatCompletionToolCallDelta
		start.Index, start.ID, start.Type = 0, call.ID, call.Type
		start.Function.Name = call.Function.Name
		var args proxias.ChatCompletionToolCallDelta
		args.Function.Arguments = call.Function.Arguments
		for _, c := range []proxias.ChatCompletionChunk{
			chunk(proxias.ChatCompletionDelta{Role: proxias.ChatMessageRoleAssistant}, nil),
			chunk(proxias.ChatCompletionDelta{ToolCalls: []proxias.ChatCompletionToolCallDelta{start}}, nil),
			chunk(proxias.ChatCompletionDelta{ToolCalls: []proxias.ChatCompletionToolCallDelta{args}}, nil),
			chunk(proxias.ChatCompletionDelta{}, &finish),
		} {
			if err := emit(c); err != nil {
				return err
			}
		}
		return nil
	}), nil
}
