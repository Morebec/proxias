package proxias_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func mockReasoningEngine(resp proxias.ChatCompletionResponse, err error) proxias.ReasoningEngine {
	return proxias.ReasoningEngineFunc(func(context.Context, proxias.ChatCompletionRequest, []proxias.ActionDescriptor) (proxias.ChatCompletionResponse, error) {
		return resp, err
	})
}

func recordingReasoningEngine(recordedReq *proxias.ChatCompletionRequest, recordedDescs *[]proxias.ActionDescriptor) proxias.ReasoningEngine {
	return proxias.ReasoningEngineFunc(func(_ context.Context, req proxias.ChatCompletionRequest, descs []proxias.ActionDescriptor) (proxias.ChatCompletionResponse, error) {
		*recordedReq = req
		*recordedDescs = descs
		return someResponse, nil
	})
}

func TestReasoningEngineFunc(t *testing.T) {
	// GIVEN
	expectedDescs := []proxias.ActionDescriptor{
		{Name: "SomeAction"},
	}
	expectedContext := t.Context()
	expectedReq := proxias.ChatCompletionRequest{Model: "some_model"}
	expectedErr := assert.AnError
	expectedResponse := proxias.ChatCompletionResponse{
		Status: 10,
	}

	reasoningEngine := proxias.ReasoningEngineFunc(func(ctx context.Context, req proxias.ChatCompletionRequest, descs []proxias.ActionDescriptor) (proxias.ChatCompletionResponse, error) {
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

func TestNewReasoningMiddleware(t *testing.T) {
	tests := []struct {
		name        string
		given       proxias.ReasoningEngine
		errContains string
	}{
		{
			name:  "canonical",
			given: mockReasoningEngine(someResponse, nil),
		},
		{
			name:        "GIVEN no engine THEN it fails",
			errContains: "needs a ReasoningEngine",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := proxias.NewReasoningMiddleware(tt.given)

			if tt.errContains != "" {
				assert.ErrorContains(t, err, tt.errContains)
				assert.Nil(t, got)
				return
			}
			require.NoError(t, err)
			assert.NotNil(t, got)
		})
	}
}

func TestReasoningMiddlewareGivesTheEngineTheRequestAndTheDescriptorsOfTheContext(t *testing.T) {
	// Given
	var recordedReq proxias.ChatCompletionRequest
	var recordedDescs []proxias.ActionDescriptor
	middleware, err := proxias.NewReasoningMiddleware(recordingReasoningEngine(&recordedReq, &recordedDescs))
	require.NoError(t, err)
	ctx := proxias.WithActionDescriptors(t.Context(), []proxias.ActionDescriptor{someActionDescriptor})

	// When it ends the chain
	got, err := middleware(nil)(ctx, someRequest)

	// Then
	require.NoError(t, err)
	assert.Equal(t, someResponse, got)
	assert.Equal(t, someRequest, recordedReq)
	assert.Equal(t, []proxias.ActionDescriptor{someActionDescriptor}, recordedDescs)
}

func TestReasoningMiddlewareWrapsTheEnginesError(t *testing.T) {
	// Given
	middleware, err := proxias.NewReasoningMiddleware(mockReasoningEngine(proxias.ChatCompletionResponse{}, assert.AnError))
	require.NoError(t, err)

	// When
	_, err = middleware(nil)(t.Context(), someRequest)

	// Then the error says which stage failed, and the cause is found
	assert.ErrorContains(t, err, "reason: ")
	assert.ErrorIs(t, err, assert.AnError)
}
