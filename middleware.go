package proxias

import "context"

// ProcessFunc is ControlPlane.Process as a function: what a Middleware receives as its next, and what it
// returns. A ProcessFunc answers a request, or fails.
type ProcessFunc func(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error)

// Middleware is a step of the chain of a ControlPlane. It receives the rest of the chain as next, and returns
// the step itself. A step either answers the request and does not call next, which bypasses the rest of the
// chain, or it calls next with the request it wants the rest to see: a step may change the request, or add
// data to the context, before it does.
//
// A request is passed by value, so changing its fields changes only the copy that next receives. Its slices,
// maps and Extra fields are shared with the caller, so a step replaces them rather than editing them in place:
//
//	req.Messages = append([]ChatMessage{system}, req.Messages...) // not: req.Messages[0] = system
//
// A step that changes the model, for instance, is a middleware of a few lines:
//
//	func routeModel(next ProcessFunc) ProcessFunc {
//		return func(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
//			req.Model = chooseModel(req)
//			return next(ctx, req)
//		}
//	}
type Middleware func(next ProcessFunc) ProcessFunc

// workflowDescriptorsKey is the key of the workflow descriptors in a context: unexported, so that only this
// package sets them.
type workflowDescriptorsKey struct{}

// WithWorkflowDescriptors returns a context that carries the descriptors of the workflows that a request may ask
// for. The WorkflowInterceptionMiddleware sets them before the rest of the chain, so that a reasoning engine can
// tell what the application can do.
func WithWorkflowDescriptors(ctx context.Context, descs []WorkflowDescriptor) context.Context {
	return context.WithValue(ctx, workflowDescriptorsKey{}, descs)
}

// WorkflowDescriptorsFrom returns the workflow descriptors that the context carries, or nil if it carries none.
func WorkflowDescriptorsFrom(ctx context.Context) []WorkflowDescriptor {
	descs, _ := ctx.Value(workflowDescriptorsKey{}).([]WorkflowDescriptor)
	return descs
}
