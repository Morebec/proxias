// Command harness is a sandbox for the middleware chain: it serves a ControlPlane as the OpenAI chat completions
// endpoint, so that a real harness (OpenClaw, see `task sandbox:openclaw`) can talk to it. To try a middleware,
// change buildControlPlane and restart `task sandbox`.
//
// Claude is the classifier, the synthesizer and the reasoning engine, and the action is say_hello_world. The
// sandbox needs ANTHROPIC_API_KEY, from the environment or from a .env file in the working directory (see
// .env.example): the environment wins over the file.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
)

// anthropicAPIKeyEnv is the environment variable that holds the Anthropic API key of the sandbox.
const anthropicAPIKeyEnv = "ANTHROPIC_API_KEY"

// confirmRestartCallID identifies the call of ask_user that confirmRestart makes, so that its answer is recognised.
// A harness may rewrite the IDs of its calls, so the ID has only letters and digits, and its answer is matched by
// prefix: OpenClaw sends back a suffixed copy of it.
const confirmRestartCallID = "proxiasconfirmrestart"

func main() {
	if err := loadDotEnv(".env"); err != nil {
		log.Fatal(err)
	}
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
	key := os.Getenv(anthropicAPIKeyEnv)
	if key == "" {
		return nil, fmt.Errorf("%s is not set: export it, or put it in a .env file", anthropicAPIKeyEnv)
	}
	client := anthropicsdk.NewClient(option.WithAPIKey(key))

	classifier := anthropic.NewIntentClassifier(client, anthropicsdk.ModelClaudeHaiku4_5)
	synthesizer := anthropic.NewActionExecutionSynthesizer(client, anthropicsdk.ModelClaudeHaiku4_5)
	actionEngine := actionEngine{}

	interception, err := proxias.NewActionInterceptionMiddleware(classifier, actionEngine, synthesizer)
	if err != nil {
		return nil, err
	}

	reasoningEngine := anthropic.NewReasoningEngine(client, anthropicsdk.ModelClaudeHaiku4_5)
	reasoning, err := proxias.NewReasoningMiddleware(reasoningEngine)
	if err != nil {
		return nil, err
	}

	return proxias.NewControlPlane(logRequests(), confirmRestart(), interception, reasoning)
}

// loadDotEnv sets the variables of a .env file that are not already in the environment, so that a variable
// exported in the shell overrides the file. A missing file is not an error.
func loadDotEnv(path string) error {
	file, err := os.Open(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name, value, ok := strings.Cut(strings.TrimPrefix(line, "export "), "=")
		if !ok {
			return fmt.Errorf("%s: %q is not NAME=value", path, line)
		}
		name = strings.TrimSpace(name)
		if _, set := os.LookupEnv(name); set {
			continue
		}
		if err := os.Setenv(name, strings.Trim(strings.TrimSpace(value), `"'`)); err != nil {
			return err
		}
	}
	return scanner.Err()
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

// actionEngine is the ActionExecutionEngine of the sandbox: one action.
type actionEngine struct{}

func (actionEngine) ActionDescriptors(context.Context, proxias.ChatCompletionRequest) ([]proxias.ActionDescriptor, error) {
	return []proxias.ActionDescriptor{{ID: "say_hello_world", Name: "Say hello world", Description: "Says hello to the world"}}, nil
}

func (actionEngine) ExecuteActions(_ context.Context, _ proxias.ChatCompletionRequest, executions []proxias.ActionExecutionIntent) ([]proxias.ActionExecutionResult, error) {
	results := make([]proxias.ActionExecutionResult, 0, len(executions))
	for _, execution := range executions {
		results = append(results, proxias.ActionExecutionResult{ActionID: execution.ActionID, Status: "completed", Data: map[string]any{"message": "Hello, World!"}})
	}
	return results, nil
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
				return proxias.NewChatCompletionResponse(req, proxias.NewChatCompletionBuilder(req).BuildText(restartAnswer(req.Messages[n-1].Content.Text())))
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

// askUser answers with a call of the ask_user tool, asking the user to confirm the restart.
func askUser(req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
	arguments := `{"questions":[{"id":"confirm_restart","header":"Restart","question":"Restart the api service now?","options":[{"label":"Yes, restart (Recommended)","description":"Restart the api service"},{"label":"No","description":"Leave it running"}]}]}`
	call := proxias.ChatToolCall{ID: confirmRestartCallID, Type: "function", Function: proxias.ChatFunctionCall{Name: "ask_user", Arguments: arguments}}
	return proxias.NewChatCompletionResponse(req, proxias.NewChatCompletionBuilder(req).BuildToolCalls(call))
}
