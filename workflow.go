package proxias

import (
	"context"
	"encoding/json"
	"fmt"
)

// Workflow is deterministic business logic: compiled code that takes a JSON input and returns a Result.
type Workflow interface {
	ID() string
	Name() string
	Description() string
	Run(ctx context.Context, input json.RawMessage) (Result, error)
}

// Candidate is what an IntentClassifier is told of a registered Workflow.
type Candidate struct {
	ID          string
	Name        string
	Description string
}

// Result is what a Workflow answered. UI is an optional, opaque dynamic UI specification for the host to
// render, Proxias never looks into it. Failed is set by the Interceptor for a Workflow that returned an
// error, Output then holding why.
type Result struct {
	WorkflowID string
	Output     json.RawMessage
	UI         json.RawMessage
	Failed     bool
}

type typedWorkflow[In, Out any] struct {
	id, name, description string
	run                   func(context.Context, In) (Out, error)
}

// Typed makes a Workflow of a strongly typed function: the input is decoded into In, the returned Out is
// encoded as the Result's Output. An Out that implements UIProvider also fills the Result's UI.
func Typed[In, Out any](id, name, description string, run func(context.Context, In) (Out, error)) Workflow {
	return typedWorkflow[In, Out]{id, name, description, run}
}

// UIProvider is implemented by the output of a Workflow that has a dynamic UI to show.
type UIProvider interface{ UISpec() json.RawMessage }

func (w typedWorkflow[In, Out]) ID() string          { return w.id }
func (w typedWorkflow[In, Out]) Name() string        { return w.name }
func (w typedWorkflow[In, Out]) Description() string { return w.description }

func (w typedWorkflow[In, Out]) Run(ctx context.Context, input json.RawMessage) (Result, error) {
	var in In
	if len(input) > 0 {
		if err := json.Unmarshal(input, &in); err != nil {
			return Result{}, fmt.Errorf("decode the input of %s: %w", w.id, err)
		}
	}
	out, err := w.run(ctx, in)
	if err != nil {
		return Result{}, err
	}
	encoded, err := json.Marshal(out)
	if err != nil {
		return Result{}, fmt.Errorf("encode the output of %s: %w", w.id, err)
	}
	result := Result{WorkflowID: w.id, Output: encoded}
	if provider, ok := any(out).(UIProvider); ok {
		result.UI = provider.UISpec()
	}
	return result, nil
}
