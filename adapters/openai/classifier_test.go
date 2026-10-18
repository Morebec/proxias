package openai_test

import (
	"encoding/json"
	"net/http"
	"testing"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

// sentInput is the input that a component sent the model: the content of the user message.
func sentInput(t *testing.T, got *sent) string {
	t.Helper()
	var call struct {
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal([]byte(got.body), &call))
	require.Len(t, call.Messages, 2)
	return call.Messages[1].Content
}

func TestIntentClassifierClassify(t *testing.T) {
	type expect struct {
		intent proxias.Intent
		err    assert.ErrorAssertionFunc
	}
	both := []proxias.ActionDescriptor{restartService, appVersion}
	restart := proxias.ActionExecutionIntent{ActionID: "restart_service", Params: map[string]any{"service": "api"}}
	tests := []struct {
		name  string
		given string // the text the model writes
		when  []proxias.ActionDescriptor
		then  expect
	}{
		{
			name:  "canonical",
			given: `{"executions":[{"action_id":"restart_service","params":{"service":"api"}}]}`,
			when:  both,
			then:  expect{intent: proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{restart}}, err: assert.NoError},
		},
		{
			name:  "GIVEN an action asked for several times THEN each is an execution, in order",
			given: `{"executions":[{"action_id":"restart_service","params":{"service":"api"}},{"action_id":"restart_service","params":{"service":"db"}}]}`,
			when:  both,
			then: expect{
				intent: proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{
					restart,
					{ActionID: "restart_service", Params: map[string]any{"service": "db"}},
				}},
				err: assert.NoError,
			},
		},
		{
			name:  "GIVEN no execution THEN the intent is a reasoning intent",
			given: `{"executions":[]}`,
			when:  both,
			then:  expect{err: assert.NoError},
		},
		{
			name:  "GIVEN an action that was not offered THEN it is dropped",
			given: `{"executions":[{"action_id":"rm_rf"}]}`,
			when:  both,
			then:  expect{err: assert.NoError},
		},
		{
			name:  "GIVEN no action THEN the API is not called and the intent is a reasoning intent",
			given: `{"executions":[]}`,
			when:  nil,
			then:  expect{err: assert.NoError},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, got := mockOpenAI(t, http.StatusOK, answerWith(tt.given))
			classifier := openai.NewIntentClassifier(client, someModel)

			intent, err := classifier.Classify(t.Context(), userSays("restart api"), tt.when)

			tt.then.err(t, err)
			assert.Equal(t, tt.then.intent, intent)
			if len(tt.when) == 0 {
				assert.Empty(t, got.body, "the API was not called")
			}
		})
	}
}

func TestIntentClassifierClassifySendsTheConversationAndTheActionsAsJSON(t *testing.T) {
	// Given
	client, got := mockOpenAI(t, http.StatusOK, answerWith(`{"executions":[]}`))
	classifier := openai.NewIntentClassifier(client, someModel)
	req := proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{
		{Role: proxias.ChatMessageRoleSystem, Content: text("We run a hosting company")},
		{Role: proxias.ChatMessageRoleUser, Content: text("restart api")},
		{Role: proxias.ChatMessageRoleAssistant, Content: text("done")},
		{Role: proxias.ChatMessageRoleUser},
	}}

	// When
	_, err := classifier.Classify(t.Context(), req, []proxias.ActionDescriptor{restartService})

	// Then
	require.NoError(t, err)
	input := sentInput(t, got)
	var decoded struct {
		Actions []proxias.ActionDescriptor `json:"actions"`
	}
	require.NoError(t, json.Unmarshal([]byte(input), &decoded))
	assert.Equal(t, []proxias.ActionDescriptor{restartService}, decoded.Actions)
	assert.Contains(t, input, `{"role":"system","content":"We run a hosting company"}`)
	assert.Contains(t, input, `{"role":"assistant","content":"done"}`)
	assert.NotContains(t, input, `"content":""`, "a message without text is left out")
}

func TestIntentClassifierClassifyReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockOpenAI(t, http.StatusTooManyRequests, `{"error":{"type":"rate_limit_error","message":"slow down"}}`)
	classifier := openai.NewIntentClassifier(client, someModel)

	// When
	_, err := classifier.Classify(t.Context(), userSays("restart api"), []proxias.ActionDescriptor{restartService})

	// Then
	var apiErr *openaisdk.Error
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, http.StatusTooManyRequests, apiErr.StatusCode)
	assert.ErrorContains(t, err, "openai: classify")
}
