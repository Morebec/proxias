package openai_test

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

func TestActionExecutionIntentParameterExtractorExtractParameters(t *testing.T) {
	type expect struct {
		executions []proxias.ActionExecutionIntent
		err        assert.ErrorAssertionFunc
	}
	tests := []struct {
		name  string
		given string // the text the model writes
		when  []proxias.ActionDescriptor
		then  expect
	}{
		{
			name:  "canonical",
			given: `{"executions":[{"action_id":"restart_service","params":{"service":"api"}}]}`,
			when:  []proxias.ActionDescriptor{restartService},
			then:  expect{executions: []proxias.ActionExecutionIntent{{ActionID: "restart_service", Params: map[string]any{"service": "api"}}}, err: assert.NoError},
		},
		{
			name:  "GIVEN an action that was not matched THEN it is dropped",
			given: `{"executions":[{"action_id":"app_version"}]}`,
			when:  []proxias.ActionDescriptor{restartService},
			then:  expect{err: assert.NoError},
		},
		{
			name:  "GIVEN no action matched THEN the API is not called",
			given: `{"executions":[{"action_id":"restart_service"}]}`,
			when:  nil,
			then:  expect{err: assert.NoError},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, got := mockOpenAI(t, http.StatusOK, answerWith(tt.given))
			extractor := openai.NewActionExecutionIntentParameterExtractor(client, someModel)

			executions, err := extractor.ExtractParameters(t.Context(), userSays("restart api"), tt.when)

			tt.then.err(t, err)
			assert.Equal(t, tt.then.executions, executions)
			if len(tt.when) == 0 {
				assert.Empty(t, got.body, "the API was not called")
			}
		})
	}
}

func TestActionExecutionIntentParameterExtractorExtractParametersSendsTheSchemas(t *testing.T) {
	// Given
	client, got := mockOpenAI(t, http.StatusOK, answerWith(`{"executions":[]}`))
	extractor := openai.NewActionExecutionIntentParameterExtractor(client, someModel)

	// When
	_, err := extractor.ExtractParameters(t.Context(), userSays("restart api"), []proxias.ActionDescriptor{restartService})

	// Then
	require.NoError(t, err)
	assert.Contains(t, sentInput(t, got), `"schema":{"required":["service"],"type":"object"}`, "the extractor sees the parameters")
}

func TestActionExecutionIntentParameterExtractorExtractParametersReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockOpenAI(t, http.StatusTooManyRequests, `{"error":{"type":"rate_limit_error","message":"slow down"}}`)
	extractor := openai.NewActionExecutionIntentParameterExtractor(client, someModel)

	// When
	_, err := extractor.ExtractParameters(t.Context(), userSays("restart api"), []proxias.ActionDescriptor{restartService})

	// Then
	assert.ErrorContains(t, err, "openai: extract parameters")
}
