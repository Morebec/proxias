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

func TestChatCompletionRequestKeepsTheFieldsItDoesNotName(t *testing.T) {
	// Given a request with fields at every level that the model does not name
	body := `{
		"model":"m","temperature":0.2,"response_format":{"type":"json_object"},
		"messages":[
			{"role":"user","name":"ada","content":[
				{"type":"text","text":"listen","input_audio":{"format":"wav"}},
				{"type":"image_url","image_url":{"url":"https://x/y.png","detail":"high","extra":1}}
			]},
			{"role":"assistant","reasoning_content":"hm","content":null,"tool_calls":[
				{"id":"c1","type":"function","index":0,"function":{"name":"restart","arguments":"{}","strict":true}}]}
		],
		"tools":[{"type":"function","strict":true,"function":{"name":"restart","parameters":{"type":"object"}}}]
	}`

	// When it is read and written back
	var req proxias.ChatCompletionRequest
	require.NoError(t, json.Unmarshal([]byte(body), &req))
	got, err := json.Marshal(req)

	// Then nothing is lost
	require.NoError(t, err)
	assert.JSONEq(t, body, string(got))
}

func TestChatCompletionRequestEditedFieldsWin(t *testing.T) {
	// Given a request whose model is changed by a middleware, and whose message keeps an unknown field
	var req proxias.ChatCompletionRequest
	require.NoError(t, json.Unmarshal([]byte(`{"model":"small","messages":[{"role":"user","name":"ada","content":"hi"}]}`), &req))
	req.Model = "large"
	req.Messages[0].Name = "grace"

	// When it is written
	got, err := json.Marshal(req)

	// Then the edits are sent, and the field that was not edited is kept
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"large","messages":[{"role":"user","name":"grace","content":"hi"}]}`, string(got))
}

func TestChatCompletionRequestExtraNeverOverridesATypedField(t *testing.T) {
	// Given an extra field that has the name of a typed one, which the typed field replaces
	req := proxias.ChatCompletionRequest{Model: "large", Messages: []proxias.ChatMessage{{Role: proxias.ChatMessageRoleUser}}, Extra: map[string]json.RawMessage{"model": json.RawMessage(`"small"`), "seed": json.RawMessage(`7`)}}

	// When it is written
	got, err := json.Marshal(req)

	// Then the typed field is sent, and the other extra is
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"large","messages":[{"role":"user","content":null}],"seed":7}`, string(got))
}

func TestChatCompletionRequestWithoutOptionalFieldsGainsNone(t *testing.T) {
	// Given a request that does not stream, nor names a model
	var req proxias.ChatCompletionRequest
	require.NoError(t, json.Unmarshal([]byte(`{"messages":[{"role":"user","content":"hi"}]}`), &req))

	// When it is written
	got, err := json.Marshal(req)

	// Then it gains no stream, nor stream options, which OpenAI refuses without a stream
	require.NoError(t, err)
	assert.JSONEq(t, `{"messages":[{"role":"user","content":"hi"}]}`, string(got))
	assert.Nil(t, req.Extra)
}

func TestChatContentWithAnExtraIsNotACollapsedString(t *testing.T) {
	// Given a lone text part that carries a field of its own
	content := proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: "listen", Extra: map[string]json.RawMessage{"input_audio": json.RawMessage(`{"format":"wav"}`)}}}

	// When it is written
	got, err := json.Marshal(content)

	// Then it stays a part, so the field is not lost
	require.NoError(t, err)
	assert.JSONEq(t, `[{"type":"text","text":"listen","input_audio":{"format":"wav"}}]`, string(got))
}
