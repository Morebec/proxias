package proxias

import (
	"context"
	"fmt"
)

// NewReasoningMiddleware makes the middleware that answers the requests no workflow fits: the reasoning engine
// is given the request and the workflow descriptors of the context. It always answers, so it ends a chain: it
// does not call next. It returns an error if the engine is nil.
func NewReasoningMiddleware(engine ReasoningEngine) (Middleware, error) {
	if engine == nil {
		return nil, fmt.Errorf("proxias: NewReasoningMiddleware needs a ReasoningEngine")
	}

	return func(ProcessFunc) ProcessFunc {
		return func(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
			response, err := engine.Reason(ctx, req, WorkflowDescriptorsFrom(ctx))
			if err != nil {
				return ChatCompletionResponse{}, fmt.Errorf("proxias: reason: %w", err)
			}
			return response, nil
		}
	}, nil
}
