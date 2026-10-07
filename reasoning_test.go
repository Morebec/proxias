package proxias_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias"
)

func TestReasoningEngineFunc(t *testing.T) {
	// GIVEN
	expectedDescs := []proxias.WorkflowDescriptor{
		{Name: "SomeWorkflow"},
	}
	expectedContext := t.Context()
	expectedReq := proxias.CompletionRequest{Model: "some_model"}
	expectedErr := assert.AnError
	expectedResponse := proxias.CompletionResponse{
		Status: 10,
	}

	reasoningEngine := proxias.ReasoningEngineFunc(func(ctx context.Context, req proxias.CompletionRequest, descs []proxias.WorkflowDescriptor) (proxias.CompletionResponse, error) {
		assert.Equal(t, expectedContext, ctx)
		assert.Equal(t, expectedReq, req)
		assert.Equal(t, expectedDescs, descs)

		return expectedResponse, expectedErr
	})

	// WHEN
	got, err := reasoningEngine.Reason(expectedContext, expectedReq, expectedDescs)

	// THEN
	assert.ErrorIs(t, err, expectedErr)
	assert.Equal(t, expectedResponse, got)
}
