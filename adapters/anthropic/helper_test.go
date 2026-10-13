package anthropic_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/adapters/anthropic"
)

func TestClaudeHelperAsk(t *testing.T) {
	// Given
	client, sent := mockAnthropic(t, 200, answerWith("hi"))
	helper := anthropic.NewClaudeHelper(client, someModel)

	// When
	got, err := helper.Ask(t.Context(), "follow these", map[string]int{"a": 1})

	// Then the instructions are a cached system block, and the input the JSON of the only user turn
	require.NoError(t, err)
	assert.Equal(t, "hi", got.Content[0].Text)
	var call struct {
		Model     string `json:"model"`
		MaxTokens int    `json:"max_tokens"`
		System    []struct {
			Text         string            `json:"text"`
			CacheControl map[string]string `json:"cache_control"`
		} `json:"system"`
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal([]byte(*sent), &call))
	assert.Equal(t, someModel, call.Model)
	assert.Equal(t, 4096, call.MaxTokens)
	require.Len(t, call.System, 1)
	assert.Equal(t, "follow these", call.System[0].Text)
	assert.Equal(t, "ephemeral", call.System[0].CacheControl["type"], "the instructions never vary: they can be cached")
	require.Len(t, call.Messages, 1)
	assert.Equal(t, "user", call.Messages[0].Role)
	assert.Equal(t, `{"a":1}`, call.Messages[0].Content[0].Text)
}

func TestClaudeHelperAskFailsForAnInputThatIsNotJSON(t *testing.T) {
	// Given
	client, sent := mockAnthropic(t, 200, answerWith("hi"))
	helper := anthropic.NewClaudeHelper(client, someModel)

	// When
	_, err := helper.Ask(t.Context(), "follow these", make(chan int))

	// Then
	assert.ErrorContains(t, err, "encode the input")
	assert.Empty(t, *sent, "the API was not called")
}

func TestClaudeHelperAskJSON(t *testing.T) {
	type answer struct {
		Value string `json:"value"`
	}
	tests := []struct {
		name  string
		given string // the API's answer
		then  assert.ErrorAssertionFunc
	}{
		{
			name:  "canonical",
			given: answerWith(`{"value":"x"}`),
			then:  assert.NoError,
		},
		{
			name:  "GIVEN the JSON is in a code fence with words around it THEN it is read",
			given: answerWith("Sure:\n```json\n{\"value\":\"x\"}\n```"),
			then:  assert.NoError,
		},
		{
			name:  "GIVEN the text is in several blocks, among others THEN the text blocks are read together",
			given: `{"id":"msg_1","model":"claude-x","stop_reason":"end_turn","content":[{"type":"text","text":"{\"value\":"},{"type":"thinking","thinking":"hm","signature":"s"},{"type":"text","text":"\"x\"}"}],"usage":{}}`,
			then:  assert.NoError,
		},
		{
			name:  "GIVEN an answer that is not JSON THEN it fails",
			given: answerWith("x"),
			then:  assert.Error,
		},
		{
			name:  "GIVEN a JSON of the wrong shape THEN it fails",
			given: answerWith(`{"value":1}`),
			then:  assert.Error,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, _ := mockAnthropic(t, 200, tt.given)
			helper := anthropic.NewClaudeHelper(client, someModel)
			var got answer

			err := helper.AskJSON(t.Context(), "follow these", struct{}{}, &got)

			tt.then(t, err)
			if err == nil {
				assert.Equal(t, answer{Value: "x"}, got)
			}
		})
	}
}
