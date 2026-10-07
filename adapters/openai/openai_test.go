package openai_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

// failingReader is a body that cannot be read, as a connection that drops mid-request is.
type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("connection reset") }

func TestDecodeRequest(t *testing.T) {
	type expect struct {
		stream   bool
		model    string
		messages []proxias.CompletionMessage
	}
	tests := []struct {
		name  string
		given string // the body
		then  expect
	}{
		{
			name:  "GIVEN a string content THEN it becomes plain text",
			given: `{"model":"gpt-x","stream":true,"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"restart \"api\"\nplease"}]}`,
			then: expect{
				stream: true,
				model:  "gpt-x",
				messages: []proxias.CompletionMessage{
					{Role: proxias.CompletionMessageRoleSystem, Content: []byte("be brief")},
					{Role: proxias.CompletionMessageRoleUser, Content: []byte("restart \"api\"\nplease")},
				},
			},
		},
		{
			name:  "GIVEN a list of content parts THEN the text parts are joined by new lines and the others are left out",
			given: `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"what is"},{"type":"image_url","image_url":{"url":"http://x/y.png"}},{"type":"text","text":"this?"}]}]}`,
			then: expect{
				model: "m",
				messages: []proxias.CompletionMessage{
					{Role: proxias.CompletionMessageRoleUser, Content: []byte("what is\nthis?")},
				},
			},
		},
		{
			name:  "GIVEN an assistant that only calls tools, a null content, THEN it is no text",
			given: `{"model":"m","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1"}]},{"role":"tool","content":"restarted","tool_call_id":"c1"}]}`,
			then: expect{
				model: "m",
				messages: []proxias.CompletionMessage{
					{Role: proxias.CompletionMessageRoleAssistant},
					{Role: proxias.CompletionMessageRoleTool, Content: []byte("restarted")},
				},
			},
		},
		{
			name:  "GIVEN a message without content THEN it is no text",
			given: `{"model":"m","messages":[{"role":"user"}]}`,
			then: expect{
				model: "m",
				messages: []proxias.CompletionMessage{
					{Role: proxias.CompletionMessageRoleUser},
				},
			},
		},
		{
			name:  "GIVEN a role that is not known THEN it is kept as it is",
			given: `{"messages":[{"role":"critic","content":"hm"}]}`,
			then: expect{
				messages: []proxias.CompletionMessage{
					{Role: "critic", Content: []byte("hm")},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := openai.DecodeRequest(strings.NewReader(tt.given))

			require.NoError(t, err)
			assert.Equal(t, tt.then.stream, got.Stream)
			assert.Equal(t, tt.then.model, got.Model)
			assert.Equal(t, tt.then.messages, got.Messages)
			assert.Equal(t, tt.given, string(got.Request), "the body, byte for byte")
		})
	}
}

func TestDecodeRequestReturnsAnErrorForABodyThatIsNotARequest(t *testing.T) {
	tests := []struct {
		name  string
		given string // the body
		then  string // what the error says
	}{
		{
			name:  "GIVEN a body that is not JSON THEN the error says it could not be decoded",
			given: `restart the api`,
			then:  "openai: decode the request",
		},
		{
			name:  "GIVEN a body that is not an object THEN the error says it could not be decoded",
			given: `[]`,
			then:  "openai: decode the request",
		},
		{
			name:  "GIVEN a request without messages THEN the error says there are none",
			given: `{"model":"m"}`,
			then:  "openai: the request has no messages",
		},
		{
			name:  "GIVEN an empty list of messages THEN the error says there are none",
			given: `{"messages":[]}`,
			then:  "openai: the request has no messages",
		},
		{
			name:  "GIVEN a content that is an object THEN the error says which message it is",
			given: `{"messages":[{"role":"user","content":"ok"},{"role":"user","content":{"text":"hi"}}]}`,
			then:  "openai: message 1: unsupported content",
		},
		{
			name:  "GIVEN a content that is a number THEN the error says which message it is",
			given: `{"messages":[{"role":"user","content":12}]}`,
			then:  "openai: message 0: unsupported content",
		},
		{
			name:  "GIVEN content parts that are not objects THEN the error says they could not be decoded",
			given: `{"messages":[{"role":"user","content":["text"]}]}`,
			then:  "openai: message 0: decode the content parts",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := openai.DecodeRequest(strings.NewReader(tt.given))

			assert.ErrorContains(t, err, tt.then)
			assert.Zero(t, got)
		})
	}
}

func TestDecodeRequestReturnsAnErrorForABodyThatCannotBeRead(t *testing.T) {
	// When a body that fails to be read is decoded
	got, err := openai.DecodeRequest(failingReader{})

	// Then the error says so, and keeps the reader's own
	assert.ErrorContains(t, err, "openai: read the request")
	assert.ErrorContains(t, err, "connection reset")
	assert.Zero(t, got)
}
