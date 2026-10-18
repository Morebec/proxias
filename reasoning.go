package proxias

import (
	"context"
	"fmt"
)

// ReasoningEngineFunc is a ReasoningEngine as a function.
type ReasoningEngineFunc func(ctx context.Context, req ChatCompletionRequest, descs []ActionDescriptor) (ChatCompletionResponse, error)

// ReasoningEngine is responsible for handling open-ended reasoning tasks that are not tied to specific action executions.
// It takes a ChatCompletionRequest and produces a ChatCompletionResponse.
// The reasoning engine may use the provided action descriptors to inform its reasoning process.
//
// It is typically the host's LLM: a request that asks for no action usually goes to it as is.
//
// Error Handling: return a non-nil error if reasoning itself fails.
type ReasoningEngine interface {
	Reason(ctx context.Context, req ChatCompletionRequest, descs []ActionDescriptor) (ChatCompletionResponse, error)
}

// Reason implements ReasoningEngine.
func (f ReasoningEngineFunc) Reason(ctx context.Context, req ChatCompletionRequest, descs []ActionDescriptor) (ChatCompletionResponse, error) {
	return f(ctx, req, descs)
}

// NewReasoningMiddleware makes the middleware that answers the requests no action fits: the reasoning engine
// is given the request and the action descriptors of the context. It always answers, so it ends a chain: it
// does not call next. It returns an error if the engine is nil.
func NewReasoningMiddleware(engine ReasoningEngine) (Middleware, error) {
	if engine == nil {
		return nil, fmt.Errorf("proxias: NewReasoningMiddleware needs a ReasoningEngine")
	}

	return func(ProcessFunc) ProcessFunc {
		return func(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
			response, err := engine.Reason(ctx, req, ActionDescriptorsFrom(ctx))
			if err != nil {
				return ChatCompletionResponse{}, fmt.Errorf("reason: %w", err)
			}
			return response, nil
		}
	}, nil
}
