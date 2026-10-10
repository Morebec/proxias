// Command basic is a hello world of Proxias: a ControlPlane with a single workflow, say_hello_world, that a
// request answers without any LLM, and Claude (set ANTHROPIC_API_KEY) for what no workflow fits.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
)

const sayHelloWorld = "say_hello_world"

// classify is an IntentClassifier of one keyword, where a real one would be a fast, local model.
func classify(_ context.Context, req proxias.ChatCompletionRequest, _ []proxias.WorkflowDescriptor) (proxias.Intent, error) {
	if !strings.Contains(strings.ToLower(req.LastMessage().Content.Text()), "hello") {
		return proxias.Intent{}, nil // a reasoning intent
	}
	return proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{{WorkflowID: sayHelloWorld}}}, nil
}

// workflows is a WorkflowExecutionEngine of one workflow.
type workflows struct{}

func (workflows) WorkflowDescriptors(context.Context, proxias.ChatCompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return []proxias.WorkflowDescriptor{{
		ID:          sayHelloWorld,
		Name:        "Say hello world",
		Description: "Says hello to the world",
	}}, nil
}

func (workflows) ExecuteWorkflows(_ context.Context, _ proxias.ChatCompletionRequest, executions []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
	results := make([]proxias.WorkflowExecutionResult, 0, len(executions))
	for _, execution := range executions {
		results = append(results, proxias.WorkflowExecutionResult{
			WorkflowID: execution.WorkflowID,
			Status:     "completed",
			Data:       map[string]any{"message": "Hello, World!"},
		})
	}
	return results, nil
}

// synthesize is a WorkflowExecutionSynthesizer that says what the workflow answered, where a real one would
// make a single LLM call.
func synthesize(_ context.Context, synthesis proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
	return proxias.NewChatCompletionResponse(proxias.ChatCompletion{
		ID:      "hello",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "proxias",
		Choices: []proxias.ChatCompletionChoice{{
			Message: proxias.ChatMessage{
				Role:    proxias.ChatMessageRoleAssistant,
				Content: proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: fmt.Sprint(synthesis.ExecutionResults[0].Data["message"])}},
			},
			FinishReason: proxias.ChatFinishReasonStop,
		}},
	})
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

	interception, err := proxias.NewWorkflowInterceptionMiddleware(
		proxias.IntentClassifierFunc(classify),
		workflows{},
		proxias.WorkflowExecutionSynthesizerFunc(synthesize),
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
