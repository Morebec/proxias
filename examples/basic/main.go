// Command basic is a hello world of Proxias: a ControlPlane with a single action, say_hello_world, that a
// request answers without any LLM, and Claude (set ANTHROPIC_API_KEY) for what no action fits.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
)

const sayHelloWorld = "say_hello_world"

// classify is an IntentClassifier of one keyword, where a real one would be a fast, local model.
func classify(_ context.Context, req proxias.ChatCompletionRequest, _ []proxias.ActionDescriptor) (proxias.Intent, error) {
	if !strings.Contains(strings.ToLower(req.LastMessage().Content.Text()), "hello") {
		return proxias.Intent{}, nil // a reasoning intent
	}
	return proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{{ActionID: sayHelloWorld}}}, nil
}

// actions is an ActionExecutionEngine of one action.
type actions struct{}

func (actions) ActionDescriptors(context.Context, proxias.ChatCompletionRequest) ([]proxias.ActionDescriptor, error) {
	return []proxias.ActionDescriptor{{
		ID:          sayHelloWorld,
		Name:        "Say hello world",
		Description: "Says hello to the world",
	}}, nil
}

func (actions) ExecuteActions(_ context.Context, _ proxias.ChatCompletionRequest, executions []proxias.ActionExecutionIntent) ([]proxias.ActionExecutionResult, error) {
	results := make([]proxias.ActionExecutionResult, 0, len(executions))
	for _, execution := range executions {
		results = append(results, proxias.ActionExecutionResult{
			ActionID: execution.ActionID,
			Status:   "completed",
			Data:     map[string]any{"message": "Hello, World!"},
		})
	}
	return results, nil
}

// synthesize is an ActionExecutionSynthesizer that says what the action answered, where a real one would
// make a single LLM call.
func synthesize(_ context.Context, synthesis proxias.ActionExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
	text := fmt.Sprint(synthesis.ExecutionResults[0].Data["message"])
	return proxias.NewChatCompletionResponse(synthesis.Request, proxias.NewChatCompletionBuilder(synthesis.Request).BuildText(text))
}

func main() {
	if err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(out io.Writer) error {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		return errors.New("set ANTHROPIC_API_KEY to run this example")
	}

	interception, err := proxias.NewActionInterceptionMiddleware(
		proxias.IntentClassifierFunc(classify),
		actions{},
		proxias.ActionExecutionSynthesizerFunc(synthesize),
	)
	if err != nil {
		return err
	}

	// The client reads ANTHROPIC_API_KEY. The key, the timeout and the retries are the host's to set.
	reasoning, err := proxias.NewReasoningMiddleware(anthropic.NewReasoningEngine(anthropicsdk.NewClient(), anthropicsdk.ModelClaudeHaiku4_5))
	if err != nil {
		return err
	}

	controlPlane, err := proxias.NewControlPlane(interception, reasoning)
	if err != nil {
		return err
	}

	for _, message := range []string{"Say hello to the world", "Write me a haiku"} {
		response, err := controlPlane.Process(context.Background(), proxias.ChatCompletionRequest{
			Messages: []proxias.ChatMessage{{
				Role:    proxias.ChatMessageRoleUser,
				Content: proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: message}},
			}},
		})
		if err != nil {
			return fmt.Errorf("%q: %w", message, err)
		}
		body, err := io.ReadAll(response.Body)
		_ = response.Body.Close()
		if err != nil {
			return fmt.Errorf("%q: read the response: %w", message, err)
		}
		fmt.Fprintf(out, "%-24s → %s\n", message, body)
	}
	return nil
}
