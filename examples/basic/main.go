// Command basic wires a toy IntentClassifier, WorkflowExecutionEngine, WorkflowExecutionSynthesizer and
// ReasoningEngine into a proxias.ControlPlane, and sends it a request of each kind: one workflow, a workflow
// that cannot run, several workflows at once, and a request no workflow fits.
package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/morebec/proxias"
)

// The vocabulary of this engine's statuses. Each engine has its own.
const (
	statusCompleted = "completed"
	statusFailed    = "failed"
)

var (
	restartService = proxias.WorkflowDescriptor{
		ID:          "restart_service",
		Name:        "Restart a service",
		Description: "Restarts a service",
		InputSchema: map[string]any{"type": "object", "required": []string{"service"}},
	}
	appVersion = proxias.WorkflowDescriptor{
		ID:          "app_version",
		Name:        "App version",
		Description: "Tells the running version",
	}
)

// keywordSelector is the first stage of the classifier: it only says which workflows a request asks for.
// A real one would be a fast, local model.
type keywordSelector struct{}

func (keywordSelector) Select(_ context.Context, req proxias.CompletionRequest, descs []proxias.WorkflowDescriptor) ([]proxias.WorkflowDescriptor, error) {
	text := lastMessageContent(req)
	var selected []proxias.WorkflowDescriptor
	for _, desc := range descs {
		if strings.Contains(text, keywordOf(desc)) {
			selected = append(selected, desc)
		}
	}
	return selected, nil
}

func keywordOf(desc proxias.WorkflowDescriptor) string {
	if desc.ID == restartService.ID {
		return "restart"
	}
	return "version"
}

// serviceExtractor is the second stage of the classifier: it reads the parameters of the selected workflows.
// A real one would ask an LLM for a structured output, giving it each selected workflow's InputSchema.
type serviceExtractor struct{}

func (serviceExtractor) ExtractParameters(_ context.Context, req proxias.CompletionRequest, matched []proxias.WorkflowDescriptor) ([]proxias.WorkflowExecutionIntent, error) {
	executions := make([]proxias.WorkflowExecutionIntent, 0, len(matched))
	for _, desc := range matched {
		execution := proxias.WorkflowExecutionIntent{WorkflowID: desc.ID}
		if desc.ID == restartService.ID {
			execution.Params = map[string]any{"service": wordAfter(lastMessageContent(req), "restart")}
		}
		executions = append(executions, execution)
	}
	return executions, nil
}

// wordAfter is the word that follows keyword in text, or "" if there is none, or if it is a conjunction.
func wordAfter(text, keyword string) string {
	words := strings.Fields(text)
	for i, word := range words {
		if word == keyword && i+1 < len(words) && words[i+1] != "and" {
			return words[i+1]
		}
	}
	return ""
}

func lastMessageContent(req proxias.CompletionRequest) string {
	return strings.ToLower(string(req.LastMessage().Content))
}

// workflows is a WorkflowExecutionEngine of two workflows.
type workflows struct{}

func (workflows) WorkflowDescriptors(context.Context, proxias.CompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return []proxias.WorkflowDescriptor{restartService, appVersion}, nil
}

func (workflows) ExecuteWorkflows(_ context.Context, _ proxias.CompletionRequest, executions []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
	results := make([]proxias.WorkflowExecutionResult, 0, len(executions))
	for _, execution := range executions {
		switch execution.WorkflowID {
		case appVersion.ID:
			results = append(results, proxias.WorkflowExecutionResult{
				WorkflowID: execution.WorkflowID,
				Status:     statusCompleted,
				Data:       map[string]any{"version": "1.4.2"},
			})
		case restartService.ID:
			results = append(results, restart(execution))
		default:
			// Not an error of the engine: the user asked for a workflow that does not exist, and hears of it.
			results = append(results, proxias.WorkflowExecutionResult{
				WorkflowID: execution.WorkflowID,
				Status:     statusFailed,
				Data:       map[string]any{"error": "there is no such workflow"},
			})
		}
	}
	return results, nil
}

// restart is a workflow that cannot run without its parameter. That is not an error of the engine: the
// result says why, and the synthesizer tells the user.
func restart(execution proxias.WorkflowExecutionIntent) proxias.WorkflowExecutionResult {
	service, _ := execution.Params["service"].(string)
	if service == "" {
		return proxias.WorkflowExecutionResult{
			WorkflowID: execution.WorkflowID,
			Status:     statusFailed,
			Data:       map[string]any{"error": "the parameter service is missing"},
		}
	}
	return proxias.WorkflowExecutionResult{
		WorkflowID: execution.WorkflowID,
		Status:     statusCompleted,
		Data:       map[string]any{"restarted": service},
	}
}

// synthesize is a WorkflowExecutionSynthesizer of fixed sentences, where a real one would make a single LLM call.
func synthesize(_ context.Context, synthesis proxias.WorkflowExecutionSynthesis) (proxias.CompletionResponse, error) {
	sentences := make([]string, 0, len(synthesis.ExecutionResults))
	for _, result := range synthesis.ExecutionResults {
		if result.Status == statusFailed {
			sentences = append(sentences, fmt.Sprintf("%s failed: %v.", result.WorkflowID, result.Data["error"]))
			continue
		}
		sentences = append(sentences, fmt.Sprintf("%s: %v.", result.WorkflowID, result.Data))
	}
	return text(strings.Join(sentences, " ")), nil
}

// reasoner is a ReasoningEngine that stands in for the LLM that answers what no workflow fits.
type reasoner struct{}

func (reasoner) Reason(_ context.Context, req proxias.CompletionRequest, _ []proxias.WorkflowDescriptor) (proxias.CompletionResponse, error) {
	return text("(an LLM would answer with a beautiful haiku)"), nil
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
	controlPlane := proxias.NewControlPlane(
		proxias.NewCompositeIntentClassifier(keywordSelector{}, serviceExtractor{}),
		workflows{},
		proxias.WorkflowExecutionSynthesizerFunc(synthesize),
		reasoner{},
	)

	messages := []string{
		"What version is running?",
		"restart api",
		"restart",
		"restart api and tell me the version",
		"Write me a haiku",
	}
	for _, message := range messages {
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
		fmt.Fprintf(out, "%-38s → %s\n", message, body)
	}
	return nil
}
