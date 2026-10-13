package openai_test

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

func TestWorkflowExecutionIntentParameterExtractorExtractParameters(t *testing.T) {
	type expect struct {
		executions []proxias.WorkflowExecutionIntent
		err        assert.ErrorAssertionFunc
	}
	tests := []struct {
		name  string
		given string // the text the model writes
		when  []proxias.WorkflowDescriptor
		then  expect
	}{
		{
			name:  "canonical",
			given: `{"executions":[{"workflow_id":"restart_service","params":{"service":"api"}}]}`,
			when:  []proxias.WorkflowDescriptor{restartService},
			then:  expect{executions: []proxias.WorkflowExecutionIntent{{WorkflowID: "restart_service", Params: map[string]any{"service": "api"}}}, err: assert.NoError},
		},
		{
			name:  "GIVEN a workflow that was not matched THEN it is dropped",
			given: `{"executions":[{"workflow_id":"app_version"}]}`,
			when:  []proxias.WorkflowDescriptor{restartService},
			then:  expect{err: assert.NoError},
		},
		{
			name:  "GIVEN no workflow matched THEN the API is not called",
			given: `{"executions":[{"workflow_id":"restart_service"}]}`,
			when:  nil,
			then:  expect{err: assert.NoError},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, got := mockOpenAI(t, http.StatusOK, answerWith(tt.given))
			extractor := openai.NewWorkflowExecutionIntentParameterExtractor(client, someModel)

			executions, err := extractor.ExtractParameters(t.Context(), userSays("restart api"), tt.when)

			tt.then.err(t, err)
			assert.Equal(t, tt.then.executions, executions)
			if len(tt.when) == 0 {
				assert.Empty(t, got.body, "the API was not called")
			}
		})
	}
}

func TestWorkflowExecutionIntentParameterExtractorExtractParametersSendsTheSchemas(t *testing.T) {
	// Given
	client, got := mockOpenAI(t, http.StatusOK, answerWith(`{"executions":[]}`))
	extractor := openai.NewWorkflowExecutionIntentParameterExtractor(client, someModel)

	// When
	_, err := extractor.ExtractParameters(t.Context(), userSays("restart api"), []proxias.WorkflowDescriptor{restartService})

	// Then
	require.NoError(t, err)
	assert.Contains(t, sentInput(t, got), `"schema":{"required":["service"],"type":"object"}`, "the extractor sees the parameters")
}

func TestWorkflowExecutionIntentParameterExtractorExtractParametersReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockOpenAI(t, http.StatusTooManyRequests, `{"error":{"type":"rate_limit_error","message":"slow down"}}`)
	extractor := openai.NewWorkflowExecutionIntentParameterExtractor(client, someModel)

	// When
	_, err := extractor.ExtractParameters(t.Context(), userSays("restart api"), []proxias.WorkflowDescriptor{restartService})

	// Then
	assert.ErrorContains(t, err, "openai: extract parameters")
}
