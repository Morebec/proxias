package proxias_test

import (
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func TestNewChatCompletionResponse(t *testing.T) {
	text := proxias.ChatCompletion{
		ID: "c1", Model: "m",
		Choices: []proxias.ChatCompletionChoice{{
			Message:      proxias.ChatMessage{Role: proxias.ChatMessageRoleAssistant, Content: proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: "hi"}}},
			FinishReason: proxias.ChatFinishReasonStop,
		}},
	}
	withUsage := text
	withUsage.Usage = &proxias.ChatUsage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}
	toolCall := proxias.ChatCompletion{
		ID: "c1", Model: "m",
		Choices: []proxias.ChatCompletionChoice{{
			Message:      proxias.ChatMessage{Role: proxias.ChatMessageRoleAssistant, ToolCalls: []proxias.ChatToolCall{{ID: "call1", Type: "function", Function: proxias.ChatFunctionCall{Name: "ask_user", Arguments: "{}"}}}},
			FinishReason: proxias.ChatFinishReasonToolCalls,
		}},
	}

	type args struct {
		req        proxias.ChatCompletionRequest
		completion proxias.ChatCompletion
	}
	type expect struct {
		contentType string
		body        string
	}
	tests := []struct {
		name string
		when args
		then expect
	}{
		{
			name: "canonical",
			when: args{
				req:        proxias.ChatCompletionRequest{},
				completion: text,
			},
			then: expect{
				contentType: "application/json",
				body:        `{"id":"c1","object":"","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`,
			},
		},
		{
			name: "WHEN the request streams THEN the completion is streamed",
			when: args{
				req:        proxias.ChatCompletionRequest{Stream: true},
				completion: text,
			},
			then: expect{
				contentType: "text/event-stream",
				body: `data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}` + "\n\n" +
					`data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
					"data: [DONE]\n\n",
			},
		},
		{
			name: "WHEN the request streams and asks for the usage THEN the usage comes last",
			when: args{
				req:        proxias.ChatCompletionRequest{Stream: true, StreamOptions: &proxias.ChatStreamOptions{IncludeUsage: true}},
				completion: withUsage,
			},
			then: expect{
				contentType: "text/event-stream",
				body: `data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}` + "\n\n" +
					`data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
					`data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}` + "\n\n" +
					"data: [DONE]\n\n",
			},
		},
		{
			name: "WHEN the request streams a tool call THEN the call is streamed",
			when: args{
				req:        proxias.ChatCompletionRequest{Stream: true},
				completion: toolCall,
			},
			then: expect{
				contentType: "text/event-stream",
				body: `data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call1","type":"function","function":{"name":"ask_user","arguments":"{}"}}]},"finish_reason":null}]}` + "\n\n" +
					`data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
					"data: [DONE]\n\n",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// When
			got, err := proxias.NewChatCompletionResponse(tt.when.req, tt.when.completion)

			// Then
			require.NoError(t, err)
			defer func() { _ = got.Body.Close() }()
			assert.Equal(t, http.StatusOK, got.Status)
			assert.Equal(t, tt.then.contentType, got.Header.Get("Content-Type"))
			raw, err := io.ReadAll(got.Body)
			require.NoError(t, err)
			assert.Equal(t, tt.then.body, string(raw))
		})
	}
}
