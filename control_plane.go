package proxias

import (
	"context"
	"errors"
	"fmt"
)

// ControlPlane answers a request by running it through a chain of Middleware: the first step that answers it
// bypasses the rest. A chain of a WorkflowInterceptionMiddleware, which runs the workflows a request asks for,
// and then a ReasoningMiddleware, which answers the rest, is the usual one.
type ControlPlane interface {
	// Process answers req. An error is an infrastructure failure of one of the steps, or of the context: a
	// workflow that cannot run for a reason the user should hear of is not an error, it is a
	// WorkflowExecutionResult for the synthesizer.
	//
	// Process does not recover from panics: the host does, if it wants to.
	Process(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error)
}

// controlPlane is the ControlPlane made by NewControlPlane. It is safe for concurrent use as long as the
// middleware it was made of are.
type controlPlane struct {
	process ProcessFunc
}

// NewControlPlane makes a ControlPlane of the middleware, in the order they are given: the first is the first
// to see a request. It returns an error if there is none, or if one of them is nil.
//
// The chain is built once, here, so that a request costs only the calls of the steps it goes through.
func NewControlPlane(middlewares ...Middleware) (ControlPlane, error) {
	if len(middlewares) == 0 {
		return nil, errors.New("proxias: NewControlPlane needs a Middleware")
	}

	process := ProcessFunc(func(context.Context, ChatCompletionRequest) (ChatCompletionResponse, error) {
		return ChatCompletionResponse{}, errors.New("proxias: no middleware answered the request")
	})
	for i := len(middlewares) - 1; i >= 0; i-- {
		if middlewares[i] == nil {
			return nil, fmt.Errorf("proxias: NewControlPlane needs a Middleware, but the one at %d is nil", i)
		}
		process = middlewares[i](process)
	}

	return &controlPlane{process: process}, nil
}

// Process implements ControlPlane.
func (c *controlPlane) Process(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	if err := ctx.Err(); err != nil {
		return ChatCompletionResponse{}, err
	}
	return c.process(ctx, req)
}
