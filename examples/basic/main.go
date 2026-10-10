// Command basic is a hello world of Proxias: a ControlPlane with a single workflow, say_hello_world, whose
// classifier and synthesizer are Claude (set ANTHROPIC_API_KEY), and a stub of the reasoning engine, which
// would be the host's LLM.
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

// workflows is a WorkflowExecutionEngine of one workflow.
type workflows struct{}

func (workflows) WorkflowDescriptors(context.Context, proxias.CompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return []proxias.WorkflowDescriptor{{
		ID:          "say_hello_world",
		Name:        "Say hello world",
		Description: "Says hello to the world",
	}}, nil
}

func (workflows) ExecuteWorkflows(_ context.Context, _ proxias.CompletionRequest, executions []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
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

// encode wraps the text that Claude wrote in the wire format of the clients: here, plain text.
func encode(_ proxias.CompletionRequest, s string) proxias.CompletionResponse {
	return text(s)
}

// reason is a ReasoningEngine that stands in for the LLM that answers what no workflow fits.
func reason(context.Context, proxias.CompletionRequest, []proxias.WorkflowDescriptor) (proxias.CompletionResponse, error) {
	return text("(an LLM would answer)"), nil
}

func text(s string) proxias.CompletionResponse {
	return proxias.CompletionResponse{Status: 200, ContentType: "text/plain", Body: io.NopCloser(strings.NewReader(s))}
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

	// The client reads ANTHROPIC_API_KEY. The key, the timeout and the retries are the host's to set.
	client := anthropicsdk.NewClient()

	synthesizer, err := anthropic.NewWorkflowExecutionSynthesizer(client, anthropicsdk.ModelClaudeHaiku4_5, encode)
	if err != nil {
		return err
	}

	controlPlane, err := proxias.NewControlPlane(
		anthropic.NewIntentClassifier(client, anthropicsdk.ModelClaudeHaiku4_5),
		workflows{},
		synthesizer,
		proxias.ReasoningEngineFunc(reason),
	)
	if err != nil {
		return err
	}

	for _, message := range []string{"Say hello to the world", "Write me a haiku"} {
		response, err := controlPlane.Process(context.Background(), proxias.CompletionRequest{
			Messages: []proxias.CompletionMessage{{Role: proxias.CompletionMessageRoleUser, Content: []byte(message)}},
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
