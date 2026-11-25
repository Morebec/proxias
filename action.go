package proxias

import (
	"context"
	"fmt"
	"slices"
	"strings"
)

// ActionDescriptor describes an action to whatever chooses among them (a model,
// a classifier): what it is called, what it does and how it is called (its
// InputSchema).
type ActionDescriptor struct {
	// ID is the unique identifier of the action.
	ID string `json:"id"`
	// Name is a human friendly name, for display purposes.
	Name string `json:"name"`
	// Description is what the action does: the agent chooses among the actions
	// with it.
	Description string `json:"description"`
	// InputSchema is the JSON Schema of the action's parameters, if it has any.
	InputSchema map[string]any `json:"schema,omitempty"`
	// Deferred tells that the model is not given the action's schema up front,
	// only its ID and description in an index, and loads the schema when it
	// needs the action: for the many actions an agent rarely runs. How it loads
	// it is the LLM adapter's.
	Deferred bool `json:"deferred,omitempty"`
}

// ActionExecutionRequest asks to run an action: the payload of an action
// request.
type ActionExecutionRequest struct {
	// ActionID is the ID of the action to run.
	ActionID string `json:"action_id"`
	// Params are the values of the action's parameters, by name, as its
	// InputSchema describes them.
	Params map[string]any `json:"params,omitempty"`
}

// ActionExecutionResult is what an action answered after being executed: the
// payload of an action result.
type ActionExecutionResult struct {
	// ActionID is the action that ran.
	ActionID string `json:"action_id"`
	// ExecutionID identifies this execution of the action, to track it when it
	// runs
	// asynchronously.
	ExecutionID string `json:"execution_id,omitempty"`
	// Status tells how it went, in the executor's own vocabulary ("completed",
	// "failed", "running"...).
	Status string `json:"status"`
	// Data is the output of the action, or why it failed.
	Data map[string]any `json:"data,omitempty"`
}

// ActionExecutor is the agent's execution layer: it describes the actions the
// agent can do, and runs them. Strategies only ask it for the descriptors and
// have the agent execute what they request; they never narrow it.
//
// It returns an error when it fails itself (a registry or a queue that is down,
// an action it does not know...). An action that cannot run for a reason the
// agent should hear of is not such an error: its result has a failing Status,
// and Data that says why.
type ActionExecutor interface {
	// Descriptors describes the actions.
	Descriptors(context.Context, Session) ([]ActionDescriptor, error)

	// Execute runs the actions, and returns exactly one result per execution
	// request, in order.
	Execute(context.Context, Session, []ActionExecutionRequest) ([]ActionExecutionResult, error)
}

// SimpleAction is an action of a SimpleActionExecutor: its descriptor, and the
// function that runs it.
type SimpleAction struct {
	ActionDescriptor
	RunFn func(ctx context.Context, s Session, params map[string]any) (ActionExecutionResult, error)
}

// SimpleActionExecutor is an ActionExecutor of functions, by action ID.
type SimpleActionExecutor struct {
	Actions map[string]SimpleAction
}

// NewSimpleActionExecutor returns the executor of the actions, by their ID.
func NewSimpleActionExecutor(actions ...SimpleAction) SimpleActionExecutor {
	byID := make(map[string]SimpleAction, len(actions))
	for _, action := range actions {
		byID[action.ID] = action
	}
	return SimpleActionExecutor{Actions: byID}
}

// Descriptors describes the actions, ordered by ID so that what a model is told
// of them does not change from one turn to the next, allowing prompt caching
// for
// models that support it.
func (e SimpleActionExecutor) Descriptors(context.Context, Session) ([]ActionDescriptor, error) {
	descriptors := make([]ActionDescriptor, 0, len(e.Actions))
	for _, action := range e.Actions {
		descriptors = append(descriptors, action.ActionDescriptor)
	}
	slices.SortFunc(descriptors, func(a, b ActionDescriptor) int {
		return strings.Compare(a.ID, b.ID)
	})
	return descriptors, nil
}

// Execute runs the actions one after the other. An unknown action fails: its
// result says so, for the model to see. The error of a function is an error.
func (e SimpleActionExecutor) Execute(
	ctx context.Context,
	s Session,
	requests []ActionExecutionRequest,
) ([]ActionExecutionResult, error) {
	results := make([]ActionExecutionResult, 0, len(requests))
	for _, req := range requests {
		action, ok := e.Actions[req.ActionID]
		if !ok || action.RunFn == nil {
			results = append(results, ActionExecutionResult{
				ActionID: req.ActionID,
				Status:   "failed",
				Data:     map[string]any{"error": "unknown action"},
			})
			continue
		}

		result, err := action.RunFn(ctx, s, req.Params)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", req.ActionID, err)
		}
		results = append(results, result)
	}
	return results, nil
}
