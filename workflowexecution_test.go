package proxias_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func TestWorkflowExecutionSynthesizerFuncSynthesize(t *testing.T) {
	var synthesized proxias.WorkflowExecutionSynthesis
	var synthesizer proxias.WorkflowExecutionSynthesizer = proxias.WorkflowExecutionSynthesizerFunc(
		func(_ context.Context, synthesis proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
			synthesized = synthesis
			return someResponse, nil
		})
	synthesis := proxias.WorkflowExecutionSynthesis{
		Request:          proxias.ChatCompletionRequest{Model: "m"},
		Intent:           proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}},
		ExecutionResults: []proxias.WorkflowExecutionResult{{WorkflowID: "some_workflow", Status: "completed"}},
	}

	got, err := synthesizer.Synthesize(context.Background(), synthesis)

	require.NoError(t, err)
	assert.Equal(t, synthesis, synthesized)
	assert.Equal(t, someResponse, got)
}

func TestWorkflowExecutionSynthesizerFuncSynthesizeReturnsTheFunctionsError(t *testing.T) {
	var synthesizer proxias.WorkflowExecutionSynthesizer = proxias.WorkflowExecutionSynthesizerFunc(
		func(context.Context, proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
			return proxias.ChatCompletionResponse{}, assert.AnError
		})

	got, err := synthesizer.Synthesize(context.Background(), proxias.WorkflowExecutionSynthesis{})

	assert.ErrorIs(t, err, assert.AnError)
	assert.Zero(t, got)
}

// NewWorkflowInterceptionMiddleware makes the middleware that runs the workflows a request asks for: the
// classifier names them, the engine runs them, and the synthesizer makes the response. A request that asks for
// no workflow is passed on to the rest of the chain, with the descriptors of the workflows in its context. It
// returns an error if one of the components is nil.
func NewWorkflowInterceptionMiddleware(classifier proxias.IntentClassifier, engine proxias.WorkflowExecutionEngine, synthesizer proxias.WorkflowExecutionSynthesizer) (proxias.Middleware, error) {
	switch {
	case classifier == nil:
		return nil, fmt.Errorf("proxias: NewWorkflowInterceptionMiddleware needs an IntentClassifier")
	case engine == nil:
		return nil, fmt.Errorf("proxias: NewWorkflowInterceptionMiddleware needs a WorkflowExecutionEngine")
	case synthesizer == nil:
		return nil, fmt.Errorf("proxias: NewWorkflowInterceptionMiddleware needs a WorkflowExecutionSynthesizer")
	}

	return func(next proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			workflowDescs, err := engine.WorkflowDescriptors(ctx, req)
			if err != nil {
				return proxias.ChatCompletionResponse{}, fmt.Errorf("list workflow descriptors: %w", err)
			}

			// Without workflows there is nothing to choose amongst: no need to ask the classifier.
			var intent proxias.Intent
			if len(workflowDescs) != 0 {
				intent, err = classifier.Classify(ctx, req, workflowDescs)
				if err != nil {
					return proxias.ChatCompletionResponse{}, fmt.Errorf("classify intent: %w", err)
				}
			}

			if intent.IsReasoningIntent() {
				return next(WithWorkflowDescriptors(ctx, workflowDescs), req)
			}

			return executeAndSynthesize(ctx, req, intent, engine, synthesizer)
		}
	}, nil
}

// executeAndSynthesize runs the workflows of an intent, and synthesizes their results into the response.
func executeAndSynthesize(ctx context.Context, req proxias.ChatCompletionRequest, intent proxias.Intent, engine proxias.WorkflowExecutionEngine, synthesizer proxias.WorkflowExecutionSynthesizer) (proxias.ChatCompletionResponse, error) {
	results, err := engine.ExecuteWorkflows(ctx, req, intent.WorkflowExecutions)
	if err != nil {
		return proxias.ChatCompletionResponse{}, fmt.Errorf("execute workflows: %w", err)
	}

	if len(results) != len(intent.WorkflowExecutions) {
		return proxias.ChatCompletionResponse{}, fmt.Errorf("execute workflows: got %d results for %d executions", len(results), len(intent.WorkflowExecutions))
	}

	// The workflows may have run past the end of the request: nobody is waiting for a response anymore.
	if err := ctx.Err(); err != nil {
		return proxias.ChatCompletionResponse{}, err
	}

	response, err := synthesizer.Synthesize(ctx, proxias.WorkflowExecutionSynthesis{
		Request:          req,
		Intent:           intent,
		ExecutionResults: results,
	})
	if err != nil {
		return proxias.ChatCompletionResponse{}, fmt.Errorf("synthesize: %w", err)
	}

	return response, nil
}

// workflowDescriptorsKey is the key of the workflow descriptors in a context: unexported, so that only this
// package sets them.
type workflowDescriptorsKey struct{}

// WithWorkflowDescriptors returns a context that carries the descriptors of the workflows that a request may ask
// for. The WorkflowInterceptionMiddleware sets them before the rest of the chain, so that a reasoning engine can
// tell what the application can do.
func WithWorkflowDescriptors(ctx context.Context, descs []proxias.WorkflowDescriptor) context.Context {
	return context.WithValue(ctx, workflowDescriptorsKey{}, descs)
}

// WorkflowDescriptorsFrom returns the workflow descriptors that the context carries, or nil if it carries none.
func WorkflowDescriptorsFrom(ctx context.Context) []proxias.WorkflowDescriptor {
	descs, _ := ctx.Value(workflowDescriptorsKey{}).([]proxias.WorkflowDescriptor)
	return descs
}
