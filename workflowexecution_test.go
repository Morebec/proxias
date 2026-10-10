package proxias_test

import (
	"context"
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
