package proxias

import "context"

// ReasoningEngine is responsible for handling open-ended reasoning tasks that are not tied to specific workflow executions.
// It takes a ChatCompletionRequest and produces a ChatCompletionResponse.
// The reasoning engine may use the provided workflow descriptors to inform its reasoning process.
//
// It is typically the host's LLM: a request that asks for no workflow usually goes to it as is.
//
// Error Handling: return a non-nil error if reasoning itself fails.
type ReasoningEngine interface {
	Reason(ctx context.Context, req ChatCompletionRequest, descs []WorkflowDescriptor) (ChatCompletionResponse, error)
}

// ReasoningEngineFunc is a ReasoningEngine as a function.
type ReasoningEngineFunc func(ctx context.Context, req ChatCompletionRequest, descs []WorkflowDescriptor) (ChatCompletionResponse, error)

// Reason implements ReasoningEngine.
func (f ReasoningEngineFunc) Reason(ctx context.Context, req ChatCompletionRequest, descs []WorkflowDescriptor) (ChatCompletionResponse, error) {
	return f(ctx, req, descs)
}
