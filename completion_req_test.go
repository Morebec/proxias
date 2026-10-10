package proxias_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func TestChatMessageHasToolRole(t *testing.T) {
	tests := []struct {
		name  string
		given proxias.ChatMessage
		then  assert.BoolAssertionFunc
	}{
		{
			name:  "GIVEN a message without a role THEN it should return false",
			given: proxias.ChatMessage{},
			then:  assert.False,
		},
		{
			name:  "GIVEN a user message THEN it should return false",
			given: proxias.ChatMessage{Role: proxias.ChatMessageRoleUser},
			then:  assert.False,
		},
		{
			name:  "GIVEN an assistant message THEN it should return false",
			given: proxias.ChatMessage{Role: proxias.ChatMessageRoleAssistant},
			then:  assert.False,
		},
		{
			name:  "GIVEN a system message THEN it should return false",
			given: proxias.ChatMessage{Role: proxias.ChatMessageRoleSystem},
			then:  assert.False,
		},
		{
			name:  "GIVEN a developer message THEN it should return false",
			given: proxias.ChatMessage{Role: proxias.ChatMessageRoleDeveloper},
			then:  assert.False,
		},
		{
			name:  "GIVEN a tool message THEN it should return true",
			given: proxias.ChatMessage{Role: proxias.ChatMessageRoleTool},
			then:  assert.True,
		},
		{
			name:  "GIVEN a function message, the former tool role, THEN it should return true",
			given: proxias.ChatMessage{Role: proxias.ChatMessageRoleFunction},
			then:  assert.True,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.then(t, tt.given.HasToolRole())
		})
	}
}

func TestChatCompletionRequestLastMessage(t *testing.T) {
	// The messages differ by their role only: LastMessage answers which one it is, not what it says.
	var (
		userRoleMessage      = proxias.ChatMessage{Role: proxias.ChatMessageRoleUser}
		assistantRoleMessage = proxias.ChatMessage{Role: proxias.ChatMessageRoleAssistant}
	)

	tests := []struct {
		name  string
		given []proxias.ChatMessage
		then  proxias.ChatMessage
	}{
		{
			name:  "GIVEN no message THEN should return no message",
			given: nil,
			then:  proxias.ChatMessage{},
		},
		{
			name:  "GIVEN one message THEN should return that message",
			given: []proxias.ChatMessage{userRoleMessage},
			then:  userRoleMessage,
		},
		{
			name: "GIVEN several messages THEN should return the last one",
			given: []proxias.ChatMessage{
				userRoleMessage,
				assistantRoleMessage,
			},
			then: assistantRoleMessage,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := proxias.ChatCompletionRequest{Messages: tt.given}

			assert.Equal(t, tt.then, req.LastMessage())
		})
	}
}

func TestChatContentUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name  string
		given string
		then  proxias.ChatContent
		err   assert.ErrorAssertionFunc
	}{
		{
			name:  "GIVEN a string THEN it is one text part",
			given: `"hello"`,
			then:  proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: "hello"}},
			err:   assert.NoError,
		},
		{
			name:  "GIVEN a list of parts THEN each is kept, the cache breakpoint and the image included",
			given: `[{"type":"text","text":"what is","cache_control":{"type":"ephemeral"}},{"type":"image_url","image_url":{"url":"http://x/y.png"}}]`,
			then: proxias.ChatContent{
				{Type: proxias.ChatContentPartTypeText, Text: "what is", CacheControl: &proxias.ChatCacheControl{Type: "ephemeral"}},
				{Type: proxias.ChatContentPartTypeImageURL, ImageURL: &proxias.ChatImageURL{URL: "http://x/y.png"}},
			},
			err: assert.NoError,
		},
		{
			name:  "GIVEN null THEN there is no part",
			given: `null`,
			err:   assert.NoError,
		},
		{
			name:  "GIVEN an object THEN it fails",
			given: `{"text":"hi"}`,
			err:   assert.Error,
		},
		{
			name:  "GIVEN a number THEN it fails",
			given: `12`,
			err:   assert.Error,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got proxias.ChatContent

			err := json.Unmarshal([]byte(tt.given), &got)

			tt.err(t, err)
			assert.Equal(t, tt.then, got)
		})
	}
}

func TestChatContentMarshalJSON(t *testing.T) {
	tests := []struct {
		name  string
		given proxias.ChatContent
		then  string
	}{
		{
			name:  "GIVEN no part THEN it is null",
			given: nil,
			then:  `null`,
		},
		{
			name:  "GIVEN a lone text part THEN it is a string",
			given: proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: "hello"}},
			then:  `"hello"`,
		},
		{
			name:  "GIVEN a text part with a cache breakpoint THEN it is a list, or the breakpoint is lost",
			given: proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: "hello", CacheControl: &proxias.ChatCacheControl{Type: "ephemeral"}}},
			then:  `[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]`,
		},
		{
			name: "GIVEN several parts THEN it is a list",
			given: proxias.ChatContent{
				{Type: proxias.ChatContentPartTypeText, Text: "what is"},
				{Type: proxias.ChatContentPartTypeImageURL, ImageURL: &proxias.ChatImageURL{URL: "http://x/y.png"}},
			},
			then: `[{"type":"text","text":"what is"},{"type":"image_url","image_url":{"url":"http://x/y.png"}}]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(tt.given)

			require.NoError(t, err)
			assert.JSONEq(t, tt.then, string(got))
		})
	}
}

func TestChatContentText(t *testing.T) {
	// Given
	content := proxias.ChatContent{
		{Type: proxias.ChatContentPartTypeText, Text: "what is"},
		{Type: proxias.ChatContentPartTypeImageURL, ImageURL: &proxias.ChatImageURL{URL: "http://x/y.png"}},
		{Type: proxias.ChatContentPartTypeText, Text: "this?"},
	}

	// When / Then
	assert.Equal(t, "what is\nthis?", content.Text(), "the text parts, joined by new lines")
	assert.Empty(t, proxias.ChatContent(nil).Text())
}
