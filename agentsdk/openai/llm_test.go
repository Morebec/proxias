package openai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
	"github.com/morebec/proxias/agentsdk/openai"
)

func TestLLMComplete(t *testing.T) {
	tests := []struct {
		name string
		// answer is the message of the completion the API answers.
		answer   string
		when     agentsdk.LLMRequest
		then     []agentsdk.LLMChunk
		messages string
		tools    string
	}{
		{
			name:   "GIVEN a text answer THEN it is a text chunk, and the system prompt system messages",
			answer: `{"role":"assistant","content":"Hello John."}`,
			when: agentsdk.LLMRequest{
				System:   []string{"You are Ada.", "Be brief."},
				Messages: []agentsdk.LLMMessage{{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "hi"}}}},
			},
			then: []agentsdk.LLMChunk{{Text: "Hello John."}},
			messages: `[
				{"role":"system","content":"You are Ada."},
				{"role":"system","content":"Be brief."},
				{"role":"user","content":"hi"}
			]`,
		},
		{
			name: "GIVEN tool calls THEN they are tool call chunks, after the text, and the actions functions",
			answer: `{"role":"assistant","content":"An email.","tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"send_email","arguments":"{\"to\":\"joe\"}"}},
				{"id":"call_2","type":"function","function":{"name":"get_time","arguments":""}}]}`,
			when: agentsdk.LLMRequest{
				Tools: []agentsdk.ActionDescriptor{
					{ID: "send_email", Description: "Sends an email.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"to": map[string]any{"type": "string"}}}},
					{ID: "get_time", Description: "Tells the time."},
				},
				Messages: []agentsdk.LLMMessage{{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "email joe"}}}},
			},
			then: []agentsdk.LLMChunk{
				{Text: "An email."},
				{ToolCall: &agentsdk.LLMToolCall{ID: "call_1", Name: "send_email", Input: map[string]any{"to": "joe"}}},
				{ToolCall: &agentsdk.LLMToolCall{ID: "call_2", Name: "get_time"}},
			},
			messages: `[{"role":"user","content":"email joe"}]`,
			tools: `[
				{"type":"function","function":{"name":"send_email","description":"Sends an email.","parameters":{"type":"object","properties":{"to":{"type":"string"}}}}},
				{"type":"function","function":{"name":"get_time","description":"Tells the time.","parameters":{"type":"object","properties":{}}}}
			]`,
		},
		{
			name:   "GIVEN tool calls and their results THEN they are the assistant's tool calls and tool messages, thinking left out",
			answer: `{"role":"assistant","content":"Sent."}`,
			when: agentsdk.LLMRequest{Messages: []agentsdk.LLMMessage{
				{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "email joe"}}},
				{Role: agentsdk.RoleAssistant, Parts: []agentsdk.LLMPart{
					{Thinking: "An email.", Signature: "sig"},
					{Text: "Sending."},
					{ToolCall: &agentsdk.LLMToolCall{ID: "call_1", Name: "send_email", Input: map[string]any{"to": "joe"}}},
				}},
				{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{
					{ToolResult: &agentsdk.LLMToolResult{CallID: "call_1", Content: "sent"}},
					{Text: "[probe.disk from system:db-1] 93%"},
				}},
			}},
			then: []agentsdk.LLMChunk{{Text: "Sent."}},
			messages: `[
				{"role":"user","content":"email joe"},
				{"role":"assistant","content":"Sending.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"send_email","arguments":"{\"to\":\"joe\"}"}}]},
				{"role":"tool","tool_call_id":"call_1","content":"sent"},
				{"role":"user","content":"[probe.disk from system:db-1] 93%"}
			]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var request map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewDecoder(r.Body).Decode(&request)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"finish_reason":"stop","message":` + tt.answer + `}]}`))
			}))
			defer server.Close()
			client := openaisdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test"), option.WithMaxRetries(0))

			var chunks []agentsdk.LLMChunk
			for c, err := range openai.NewLLM(client, "m").Complete(context.Background(), tt.when) {
				require.NoError(t, err)
				chunks = append(chunks, c)
			}

			assert.Equal(t, tt.then, chunks)
			assert.Equal(t, "m", request["model"])
			messages, _ := json.Marshal(request["messages"])
			assert.JSONEq(t, tt.messages, string(messages))
			if tt.tools != "" {
				tools, _ := json.Marshal(request["tools"])
				assert.JSONEq(t, tt.tools, string(tools))
			} else {
				assert.Nil(t, request["tools"])
			}
		})
	}
}

func TestLLMCompleteError(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
	}{
		{name: "GIVEN a bad key THEN it is the SDK's error", status: http.StatusUnauthorized, body: `{"error":{"message":"bad key","type":"invalid_request_error"}}`},
		{name: "GIVEN a rate limit THEN it is the SDK's error", status: http.StatusTooManyRequests, body: `{"error":{"message":"slow down","type":"rate_limit_error"}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()
			client := openaisdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test"), option.WithMaxRetries(0))

			var err error
			for _, e := range openai.NewLLM(client, "m").Complete(context.Background(), agentsdk.LLMRequest{}) {
				err = e
			}

			var apiErr *openaisdk.Error
			require.ErrorAs(t, err, &apiErr)
			assert.Equal(t, tt.status, apiErr.StatusCode)
		})
	}
}
