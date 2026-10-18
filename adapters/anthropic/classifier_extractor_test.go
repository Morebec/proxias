package anthropic_test

import (
	"encoding/json"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
)

var restartIntent = proxias.ActionExecutionIntent{ActionID: "restart_service", Params: map[string]any{"service": "api"}}

func TestActionExecutionIntentParameterExtractorExtractParameters(t *testing.T) {
	type expect struct {
		executions []proxias.ActionExecutionIntent
		err        assert.ErrorAssertionFunc
	}
	tests := []struct {
		name  string
		given string // the text the model writes
		then  expect
	}{
		{
			name:  "canonical",
			given: `{"executions":[{"action_id":"restart_service","params":{"service":"api"}}]}`,
			then:  expect{executions: []proxias.ActionExecutionIntent{restartIntent}, err: assert.NoError},
		},
		{
			name:  "GIVEN a parameter that the conversation does not give THEN it is left out",
			given: `{"executions":[{"action_id":"restart_service","params":{}}]}`,
			then:  expect{executions: []proxias.ActionExecutionIntent{{ActionID: "restart_service", Params: map[string]any{}}}, err: assert.NoError},
		},
		{
			name:  "GIVEN an action that was not matched THEN it is dropped",
			given: `{"executions":[{"action_id":"app_version"}]}`,
			then:  expect{err: assert.NoError},
		},
		{
			name:  "GIVEN an answer that is not JSON THEN it fails",
			given: "the service is api",
			then:  expect{err: assert.Error},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, _ := mockAnthropic(t, 200, answerWith(tt.given))
			extractor := anthropic.NewActionExecutionIntentParameterExtractor(client, someModel)

			got, err := extractor.ExtractParameters(t.Context(), userSays("restart api"), []proxias.ActionDescriptor{restartService})

			tt.then.err(t, err)
			assert.Equal(t, tt.then.executions, got)
		})
	}
}

func TestActionExecutionIntentParameterExtractorExtractParametersSendsTheSchemas(t *testing.T) {
	// Given
	client, sent := mockAnthropic(t, 200, answerWith(`{"executions":[]}`))
	extractor := anthropic.NewActionExecutionIntentParameterExtractor(client, someModel)

	// When
	_, err := extractor.ExtractParameters(t.Context(), userSays("restart api"), []proxias.ActionDescriptor{restartService})

	// Then
	require.NoError(t, err)
	var call struct {
		System []struct {
			CacheControl map[string]string `json:"cache_control"`
		} `json:"system"`
		Messages []struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal([]byte(*sent), &call))
	assert.Equal(t, "ephemeral", call.System[0].CacheControl["type"], "the instructions never vary: they can be cached")
	user := call.Messages[0].Content[0].Text
	assert.Contains(t, user, `"schema":{"required":["service"],"type":"object"}`, "the extractor sees the parameters")
}

func TestActionExecutionIntentParameterExtractorExtractParametersWithoutDescriptorsDoesNotCallTheAPI(t *testing.T) {
	// Given
	client, sent := mockAnthropic(t, 200, answerWith(`{"executions":[]}`))

	// When
	got, err := anthropic.NewActionExecutionIntentParameterExtractor(client, someModel).ExtractParameters(t.Context(), userSays("restart api"), nil)

	// Then
	require.NoError(t, err)
	assert.Empty(t, got)
	assert.Empty(t, *sent)
}

func TestActionExecutionIntentParameterExtractorExtractParametersReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockAnthropic(t, 429, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
	extractor := anthropic.NewActionExecutionIntentParameterExtractor(client, someModel)

	// When
	_, err := extractor.ExtractParameters(t.Context(), userSays("restart api"), []proxias.ActionDescriptor{restartService})

	// Then
	var apiErr *anthropicsdk.Error
	require.ErrorAs(t, err, &apiErr)
	assert.ErrorContains(t, err, "anthropic: extract parameters")
}
