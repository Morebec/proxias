package proxias

import (
	"context"
	"fmt"
)

// ActionExecutionEngine is where the deterministic, compiled actions live: it describes them and runs
// them. An action is whatever the engine runs for a request: a plain function as much as a workflow, a
// process or a pipeline.
//
// Error Handling: implementations must return a non-nil error if the engine itself fails (a registry or a queue that is down...).
// These are infrastructure errors, and the control plane returns them: they are not synthesized into a
// response for the user.
// An action that cannot run for a reason the user should hear of is not such an error: its
// ActionExecutionResult has a failing Status, and Data that says why. That covers invalid input, missing
// required parameters, and an action ID that the engine does not know.
type ActionExecutionEngine interface {
	// ActionDescriptors describes the actions that this engine provides.
	// Implementations may narrow that list based on the specifics of the ChatCompletionRequest.
	ActionDescriptors(context.Context, ChatCompletionRequest) ([]ActionDescriptor, error)

	// ExecuteActions runs the actions, and returns exactly one result per action execution.
	ExecuteActions(context.Context, ChatCompletionRequest, []ActionExecutionIntent) ([]ActionExecutionResult, error)
}

// ActionExecutionSynthesizer translates raw execution results into human-readable responses.
// It serves as the outbound translation layer in Proxias, transforming structured execution results
// into human-readable prose or UI specs as a ChatCompletionResponse.
//
// Error Handling: return a non-nil error if synthesis itself fails to generate a ChatCompletionResponse.
type ActionExecutionSynthesizer interface {
	Synthesize(context.Context, ActionExecutionSynthesis) (ChatCompletionResponse, error)
}

// ActionExecutionSynthesizerFunc is an ActionExecutionSynthesizer as a function.
type ActionExecutionSynthesizerFunc func(context.Context, ActionExecutionSynthesis) (ChatCompletionResponse, error)

// Synthesize implements ActionExecutionSynthesizer.
func (f ActionExecutionSynthesizerFunc) Synthesize(ctx context.Context, synthesis ActionExecutionSynthesis) (ChatCompletionResponse, error) {
	return f(ctx, synthesis)
}

// ActionDescriptor describes an action to the IntentClassifier, and to the model behind it, if any: what
// it is called, what it does and how it is called (its InputSchema).
type ActionDescriptor struct {
	// ID is the unique identifier of the action.
	ID string `json:"id"`
	// Name is a human friendly name, for display purposes.
	Name string `json:"name"`
	// Description is a human friendly description of what the action does. A classifier chooses among the
	// actions with it.
	Description string `json:"description"`
	// InputSchema is the JSON Schema of the action's parameters, if it has any.
	InputSchema map[string]any `json:"schema,omitempty"`
}

// ActionExecutionResult is what an action answered after being executed.
// If an action does not provide a result, it should still return an ActionExecutionResult.
type ActionExecutionResult struct {
	// ActionID is the action that ran.
	ActionID string `json:"action_id"`
	// ExecutionID is a unique identifier for this particular execution of the action,
	// this can be used to track its status in case of asynchronous execution.
	ExecutionID string `json:"execution_id,omitempty"`
	// Status represents the current execution status of the action, indicating if it was successful or not.
	// Its vocabulary is the engine's own ("completed", "failed", "running"...): the synthesizer, which knows
	// that engine, interprets it.
	Status string `json:"status"`
	// Data contains the output of the action execution, if any.
	// This data can also describe an error if the action failed, in which case the status
	// should indicate a failure.
	Data map[string]any `json:"data,omitempty"`
}

// ActionExecutionSynthesis is what an ActionExecutionSynthesizer makes a response of.
type ActionExecutionSynthesis struct {
	Request          ChatCompletionRequest   `json:"request,omitzero"`
	Intent           Intent                  `json:"intent,omitzero"`
	ExecutionResults []ActionExecutionResult `json:"execution_results,omitempty"`
}

// NewActionInterceptionMiddleware makes the middleware that runs the actions a request asks for: the
// classifier names them, the engine runs them, and the synthesizer makes the response. A request that asks for
// no action is passed on to the rest of the chain, with the descriptors of the actions in its context. It
// returns an error if one of the components is nil.
func NewActionInterceptionMiddleware(classifier IntentClassifier, engine ActionExecutionEngine, synthesizer ActionExecutionSynthesizer) (Middleware, error) {
	switch {
	case classifier == nil:
		return nil, fmt.Errorf("proxias: NewActionInterceptionMiddleware needs an IntentClassifier")
	case engine == nil:
		return nil, fmt.Errorf("proxias: NewActionInterceptionMiddleware needs an ActionExecutionEngine")
	case synthesizer == nil:
		return nil, fmt.Errorf("proxias: NewActionInterceptionMiddleware needs an ActionExecutionSynthesizer")
	}

	return func(next ProcessFunc) ProcessFunc {
		return func(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
			actionDescs, err := engine.ActionDescriptors(ctx, req)
			if err != nil {
				return ChatCompletionResponse{}, fmt.Errorf("list action descriptors: %w", err)
			}

			// Without actions there is nothing to choose amongst: no need to ask the classifier.
			var intent Intent
			if len(actionDescs) != 0 {
				intent, err = classifier.Classify(ctx, req, actionDescs)
				if err != nil {
					return ChatCompletionResponse{}, fmt.Errorf("classify intent: %w", err)
				}
			}

			if intent.IsReasoningIntent() {
				return next(WithActionDescriptors(ctx, actionDescs), req)
			}

			return executeAndSynthesize(ctx, req, intent, engine, synthesizer)
		}
	}, nil
}

// executeAndSynthesize runs the actions of an intent, and synthesizes their results into the response.
func executeAndSynthesize(ctx context.Context, req ChatCompletionRequest, intent Intent, engine ActionExecutionEngine, synthesizer ActionExecutionSynthesizer) (ChatCompletionResponse, error) {
	results, err := engine.ExecuteActions(ctx, req, intent.ActionExecutions)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("execute actions: %w", err)
	}

	if len(results) != len(intent.ActionExecutions) {
		return ChatCompletionResponse{}, fmt.Errorf("execute actions: got %d results for %d executions", len(results), len(intent.ActionExecutions))
	}

	// The actions may have run past the end of the request: nobody is waiting for a response anymore.
	if err := ctx.Err(); err != nil {
		return ChatCompletionResponse{}, err
	}

	response, err := synthesizer.Synthesize(ctx, ActionExecutionSynthesis{
		Request:          req,
		Intent:           intent,
		ExecutionResults: results,
	})
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("synthesize: %w", err)
	}

	return response, nil
}

// actionDescriptorsKey is the key of the action descriptors in a context: unexported, so that only this
// package sets them.
type actionDescriptorsKey struct{}

// WithActionDescriptors returns a context that carries the descriptors of the actions that a request may ask
// for. The ActionInterceptionMiddleware sets them before the rest of the chain, so that a reasoning engine can
// tell what the application can do.
func WithActionDescriptors(ctx context.Context, descs []ActionDescriptor) context.Context {
	return context.WithValue(ctx, actionDescriptorsKey{}, descs)
}

// ActionDescriptorsFrom returns the action descriptors that the context carries, or nil if it carries none.
func ActionDescriptorsFrom(ctx context.Context) []ActionDescriptor {
	descs, _ := ctx.Value(actionDescriptorsKey{}).([]ActionDescriptor)
	return descs
}
