package proxias

import "context"

// WorkflowExecutionEngine is where the deterministic, compiled workflows live: it describes them and runs
// them.
//
// Error Handling: implementations must return a non-nil error if the engine itself fails (a registry or a queue that is down...).
// These are infrastructure errors, and the control plane returns them: they are not synthesized into a
// response for the user.
// A workflow that cannot run for a reason the user should hear of is not such an error: its
// WorkflowExecutionResult has a failing Status, and Data that says why. That covers invalid input, missing
// required parameters, and a workflow ID that the engine does not know.
type WorkflowExecutionEngine interface {
	// WorkflowDescriptors describes the workflows that this engine provides.
	// Implementations may narrow that list based on the specifics of the CompletionRequest.
	WorkflowDescriptors(context.Context, CompletionRequest) ([]WorkflowDescriptor, error)

	// ExecuteWorkflows runs the workflows, and returns exactly one result per workflow execution.
	ExecuteWorkflows(context.Context, CompletionRequest, []WorkflowExecutionIntent) ([]WorkflowExecutionResult, error)
}

// WorkflowExecutionSynthesizer translates raw execution results into human-readable responses.
// It serves as the outbound translation layer in Proxias, transforming structured execution results
// into human-readable prose or UI specs as a CompletionResponse.
//
// Error Handling: return a non-nil error if synthesis itself fails to generate a CompletionResponse.
type WorkflowExecutionSynthesizer interface {
	Synthesize(context.Context, WorkflowExecutionSynthesis) (CompletionResponse, error)
}

// WorkflowExecutionSynthesizerFunc is a WorkflowExecutionSynthesizer as a function.
type WorkflowExecutionSynthesizerFunc func(context.Context, WorkflowExecutionSynthesis) (CompletionResponse, error)

// Synthesize implements WorkflowExecutionSynthesizer.
func (f WorkflowExecutionSynthesizerFunc) Synthesize(ctx context.Context, synthesis WorkflowExecutionSynthesis) (CompletionResponse, error) {
	return f(ctx, synthesis)
}

// WorkflowDescriptor describes a workflow to the IntentClassifier, and to the model behind it, if any: what
// it is called, what it does and how it is called (its InputSchema).
type WorkflowDescriptor struct {
	// ID is the unique identifier of the workflow.
	ID string `json:"id"`
	// Name is a human friendly name, for display purposes.
	Name string `json:"name"`
	// Description is a human friendly description of what the workflow does. A classifier chooses among the
	// workflows with it.
	Description string `json:"description"`
	// InputSchema is the JSON Schema of the workflow's parameters, if it has any.
	InputSchema map[string]any `json:"schema,omitempty"`
}

// WorkflowExecutionResult is what a workflow answered after being executed.
// If a workflow does not provide a result, it should still return a WorkflowExecutionResult.
type WorkflowExecutionResult struct {
	// WorkflowID is the workflow that ran.
	WorkflowID string `json:"workflow_id"`
	// ExecutionID is a unique identifier for this particular execution of the workflow,
	// this can be used to track its status in case of asynchronous execution.
	ExecutionID string `json:"execution_id,omitempty"`
	// Status represents the current execution status of the workflow, indicating if it was successful or not.
	// Its vocabulary is the engine's own ("completed", "failed", "running"...): the synthesizer, which knows
	// that engine, interprets it.
	Status string `json:"status"`
	// Data contains the output of the workflow execution, if any.
	// This data can also describe an error if the workflow failed, in which case the status
	// should indicate a failure.
	Data map[string]any `json:"data,omitempty"`
}

// WorkflowExecutionSynthesis is what a WorkflowExecutionSynthesizer makes a response of.
type WorkflowExecutionSynthesis struct {
	Request          CompletionRequest         `json:"request,omitzero"`
	Intent           Intent                    `json:"intent,omitzero"`
	ExecutionResults []WorkflowExecutionResult `json:"execution_results,omitempty"`
}
