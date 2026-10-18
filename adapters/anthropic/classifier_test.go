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

var (
	restartService = proxias.ActionDescriptor{
		ID:          "restart_service",
		Name:        "Restart a service",
		Description: "Restarts a service",
		InputSchema: map[string]any{"type": "object", "required": []any{"service"}},
	}
	appVersion = proxias.ActionDescriptor{ID: "app_version", Name: "App version", Description: "Tells the running version"}
)

// answerWith is the API's answer to a message, in which the model wrote text.
func answerWith(text string) string {
	encoded, _ := json.Marshal(text)
	return `{"id":"msg_1","model":"claude-x","stop_reason":"end_turn","content":[{"type":"text","text":` + string(encoded) + `}],"usage":{}}`
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
			name:  "GIVEN an action without parameters THEN its params are nil",
			given: `{"executions":[{"action_id":"app_version"}]}`,
			when:  both,
			then:  expect{intent: proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{{ActionID: "app_version"}}}, err: assert.NoError},
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
			client, sent := mockAnthropic(t, 200, answerWith(tt.given))
			classifier := anthropic.NewIntentClassifier(client, someModel)

			got, err := classifier.Classify(t.Context(), userSays("restart api"), tt.when)

			tt.then.err(t, err)
			assert.Equal(t, tt.then.intent, got)
			if len(tt.when) == 0 {
				assert.Empty(t, *sent, "the API was not called")
			}
		})
	}
}

func TestIntentClassifierClassifySendsTheConversationAndTheActionsAsJSON(t *testing.T) {
	// Given
	client, sent := mockAnthropic(t, 200, answerWith(`{"executions":[]}`))
	classifier := anthropic.NewIntentClassifier(client, someModel)
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
	var call struct {
		Messages []struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal([]byte(*sent), &call))
	require.Len(t, call.Messages, 1)
	var input struct {
		Actions []proxias.ActionDescriptor `json:"actions"`
	}
	require.NoError(t, json.Unmarshal([]byte(call.Messages[0].Content[0].Text), &input))
	assert.Equal(t, []proxias.ActionDescriptor{restartService}, input.Actions)
	assert.Contains(t, call.Messages[0].Content[0].Text, `{"role":"system","content":"We run a hosting company"}`)
	assert.Contains(t, call.Messages[0].Content[0].Text, `{"role":"assistant","content":"done"}`)
	assert.NotContains(t, call.Messages[0].Content[0].Text, `"content":""`, "a message without text is left out")
}

func TestIntentClassifierClassifyReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockAnthropic(t, 429, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
	classifier := anthropic.NewIntentClassifier(client, someModel)

	// When
	_, err := classifier.Classify(t.Context(), userSays("restart api"), []proxias.ActionDescriptor{restartService})

	// Then
	var apiErr *anthropicsdk.Error
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, 429, apiErr.StatusCode)
	assert.ErrorContains(t, err, "anthropic: classify")
}
