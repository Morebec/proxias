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

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

// mockModelAnswering is a chat completions API that answers the messages in
// turn (the last one again once all are given), and records the requests it
// was sent.
func mockModelAnswering(t *testing.T, requests *[]map[string]any, messages ...string) openaisdk.Client {
	t.Helper()
	server := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var request map[string]any
			_ = json.NewDecoder(r.Body).Decode(&request)
			*requests = append(*requests, request)
			message := messages[min(len(*requests), len(messages))-1]
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(
				`{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":` + message + `}]}`,
			))
		}),
	)
	t.Cleanup(server.Close)
	return openaisdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test"), option.WithMaxRetries(0))
}

func TestLLMCompleteDeferredActions(t *testing.T) {
	describe := `{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"describe_actions","arguments":"{\"names\":[\"send_email\"]}"}}]}`
	run := `{"role":"assistant","content":"","tool_calls":[{"id":"call_2","type":"function","function":{"name":"run_action","arguments":"{\"name\":\"send_email\",\"params\":{\"to\":\"bob\"}}"}}]}`
	described := `{"openai_described_actions":{"call_id":"call_1","names":["send_email"],"answer":"{\"actions\":[{\"name\":\"send_email\",\"description\":\"Sends an email.\",\"parameters\":{\"type\":\"object\"}}]}"}}`
	sendEmail := proxias.ActionDescriptor{
		ID:          "send_email",
		Description: "Sends an email.",
		InputSchema: map[string]any{"type": "object"},
		Deferred:    true,
	}
	emailBob := proxias.LLMMessage{
		Role:  proxias.LLMMessageRoleUser,
		Parts: []proxias.LLMPart{{Text: "email bob"}},
	}

	type expect struct {
		chunks   []proxias.LLMChunk
		requests int
		// tools are the tools of the first request; messages the messages of
		// the last.
		tools    string
		messages string
	}
	tests := []struct {
		name    string
		answers []string
		when    proxias.LLMRequest
		then    expect
	}{
		{
			name:    "canonical",
			answers: []string{describe, run},
			when: proxias.LLMRequest{
				Messages: []proxias.LLMMessage{emailBob},
				Tools:    []proxias.ActionDescriptor{{ID: "get_time", Description: "The time."}, sendEmail},
			},
			then: expect{
				chunks: []proxias.LLMChunk{
					{Opaque: json.RawMessage(described)},
					{ToolCall: &proxias.LLMToolCall{ID: "call_2", Name: "send_email", Input: map[string]any{"to": "bob"}}},
				},
				requests: 2,
				tools: `[
					{"type":"function","function":{"name":"get_time","description":"The time.","parameters":{"type":"object","properties":{}}}},
					{"type":"function","function":{"name":"describe_actions","description":"Gives the parameters of actions to look up, by name, before you run them.","parameters":{"type":"object","properties":{"names":{"type":"array","items":{"type":"string"}}},"required":["names"]}}},
					{"type":"function","function":{"name":"run_action","description":"Runs an action to look up, described first, with its parameters.","parameters":{"type":"object","properties":{"name":{"type":"string"},"params":{"type":"object"}},"required":["name","params"]}}}
				]`,
				messages: `[
					{"role":"system","content":"## Actions to look up\nBefore you run one of these actions, get its parameters with describe_actions (several at once if you need them), then run it with run_action.\n- send_email: Sends an email."},
					{"role":"user","content":"email bob"},
					{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"describe_actions","arguments":"{\"names\":[\"send_email\"]}"}}]},
					{"role":"tool","tool_call_id":"call_1","content":"{\"actions\":[{\"name\":\"send_email\",\"description\":\"Sends an email.\",\"parameters\":{\"type\":\"object\"}}]}"}
				]`,
			},
		},
		{
			name:    "GIVEN a described action run earlier THEN the exchange and the call are given back as the model made them",
			answers: []string{`{"role":"assistant","content":"Sent."}`},
			when: proxias.LLMRequest{
				Messages: []proxias.LLMMessage{
					emailBob,
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{Opaque: json.RawMessage(described)},
							{ToolCall: &proxias.LLMToolCall{ID: "call_2", Name: "send_email", Input: map[string]any{"to": "bob"}}},
						},
					},
					{
						Role:  proxias.LLMMessageRoleTool,
						Parts: []proxias.LLMPart{{ToolResult: &proxias.LLMToolResult{CallID: "call_2", Content: "sent"}}},
					},
				},
				Tools: []proxias.ActionDescriptor{sendEmail},
			},
			then: expect{
				chunks:   []proxias.LLMChunk{{Text: "Sent."}},
				requests: 1,
				messages: `[
					{"role":"system","content":"## Actions to look up\nBefore you run one of these actions, get its parameters with describe_actions (several at once if you need them), then run it with run_action.\n- send_email: Sends an email."},
					{"role":"user","content":"email bob"},
					{"role":"assistant","tool_calls":[
						{"id":"call_1","type":"function","function":{"name":"describe_actions","arguments":"{\"names\":[\"send_email\"]}"}},
						{"id":"call_2","type":"function","function":{"name":"run_action","arguments":"{\"name\":\"send_email\",\"params\":{\"to\":\"bob\"}}"}}
					]},
					{"role":"tool","tool_call_id":"call_1","content":"{\"actions\":[{\"name\":\"send_email\",\"description\":\"Sends an email.\",\"parameters\":{\"type\":\"object\"}}]}"},
					{"role":"tool","tool_call_id":"call_2","content":"sent"}
				]`,
			},
		},
		{
			name:    "GIVEN a model that only describes THEN it is asked at most three times",
			answers: []string{describe},
			when: proxias.LLMRequest{
				Messages: []proxias.LLMMessage{emailBob},
				Tools:    []proxias.ActionDescriptor{sendEmail},
			},
			then: expect{
				chunks: []proxias.LLMChunk{
					{Opaque: json.RawMessage(described)},
					{Opaque: json.RawMessage(described)},
					{Opaque: json.RawMessage(described)},
				},
				requests: 3,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var requests []map[string]any
			llm := openai.NewLLM(mockModelAnswering(t, &requests, tt.answers...), "m")
			llm.Buffered = true

			var chunks []proxias.LLMChunk
			for c, err := range llm.Complete(context.Background(), tt.when) {
				require.NoError(t, err)
				chunks = append(chunks, c)
			}

			assert.Equal(t, tt.then.chunks, chunks)
			require.Len(t, requests, tt.then.requests)
			if tt.then.tools != "" {
				tools, _ := json.Marshal(requests[0]["tools"])
				assert.JSONEq(t, tt.then.tools, string(tools))
			}
			if tt.then.messages != "" {
				messages, _ := json.Marshal(requests[len(requests)-1]["messages"])
				assert.JSONEq(t, tt.then.messages, string(messages))
			}
		})
	}
}
