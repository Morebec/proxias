package anthropic_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
)

const someModel = "some-model"

// textMessage is the API's answer to a message.
const textMessage = `{"id":"msg_1","model":"claude-x","stop_reason":"end_turn","content":[{"type":"text","text":"Hi "},{"type":"text","text":"there"}],"usage":{"input_tokens":10,"cache_read_input_tokens":5,"output_tokens":4}}`

// mockAnthropic is a server that answers every call with status and body, as a stream if the body is one,
// and keeps the body of the last request it got.
func mockAnthropic(t *testing.T, status int, body string) (anthropicsdk.Client, *string) {
	t.Helper()

	var sent string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		sent = string(raw)

		contentType := "application/json"
		if strings.HasPrefix(body, "event:") {
			contentType = "text/event-stream"
		}
		w.Header().Set("content-type", contentType)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)

	client := anthropicsdk.NewClient(option.WithAPIKey("secret-key"), option.WithBaseURL(server.URL), option.WithMaxRetries(0))
	return client, &sent
}

// sse is the events of a stream of the Messages API, each given by its type and its data.
func sse(events ...string) string {
	var stream strings.Builder
	for _, event := range events {
		var typed struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(event), &typed)
		fmt.Fprintf(&stream, "event: %s\ndata: %s\n\n", typed.Type, event)
	}
	return stream.String()
}

// picked is the fields of the request that was sent, as JSON.
func picked(t *testing.T, sent string, fields ...string) string {
	t.Helper()
	var all map[string]json.RawMessage
	require.NoError(t, json.Unmarshal([]byte(sent), &all))
	subset := map[string]json.RawMessage{}
	for _, field := range fields {
		if value, ok := all[field]; ok {
			subset[field] = value
		}
	}
	out, err := json.Marshal(subset)
	require.NoError(t, err)
	return string(out)
}

func text(s string) proxias.ChatContent {
	return proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: s}}
}

func userSays(s string) proxias.ChatCompletionRequest {
	return proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{{Role: proxias.ChatMessageRoleUser, Content: text(s)}}}
}

func TestReasoningEngineReasonMapsTheRequest(t *testing.T) {
	no := false
	tests := []struct {
		name  string
		given proxias.ChatCompletionRequest
		then  string // the fields that were sent
	}{
		{
			name:  "canonical",
			given: userSays("hello"),
			then:  `{"model":"some-model","max_tokens":4096,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`,
		},
		{
			name: "GIVEN system and developer messages THEN each text part is a block of the system prompt, with its cache breakpoint",
			given: proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{
				{Role: proxias.ChatMessageRoleSystem, Content: proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: "We run a hosting company", CacheControl: &proxias.ChatCacheControl{Type: "ephemeral", TTL: "1h"}}}},
				{Role: proxias.ChatMessageRoleDeveloper, Content: text("Be brief")},
				{Role: proxias.ChatMessageRoleUser, Content: text("hello")},
			}},
			then: `{"system":[{"type":"text","text":"We run a hosting company","cache_control":{"type":"ephemeral","ttl":"1h"}},{"type":"text","text":"Be brief"}],"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`,
		},
		{
			name: "GIVEN a tool call and its result THEN they are tool_use and tool_result blocks, the result and what follows being one user message",
			given: proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{
				{Role: proxias.ChatMessageRoleUser, Content: text("restart api")},
				{Role: proxias.ChatMessageRoleAssistant, ToolCalls: []proxias.ChatToolCall{{ID: "c1", Type: "function", Function: proxias.ChatFunctionCall{Name: "restart", Arguments: `{"service":"api"}`}}}},
				{Role: proxias.ChatMessageRoleTool, ToolCallID: "c1", Content: text("done")},
				{Role: proxias.ChatMessageRoleUser, Content: text("thanks")},
			}},
			then: `{"messages":[
				{"role":"user","content":[{"type":"text","text":"restart api"}]},
				{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"restart","input":{"service":"api"}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"done"}]},{"type":"text","text":"thanks"}]}
			]}`,
		},
		{
			name: "GIVEN images THEN a data URL is a base64 image and an address a URL image",
			given: proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{{Role: proxias.ChatMessageRoleUser, Content: proxias.ChatContent{
				{Type: proxias.ChatContentPartTypeImageURL, ImageURL: &proxias.ChatImageURL{URL: "data:image/png;base64,AAAA"}},
				{Type: proxias.ChatContentPartTypeImageURL, ImageURL: &proxias.ChatImageURL{URL: "https://x/y.png"}},
				{Type: proxias.ChatContentPartTypeImageURL, ImageURL: &proxias.ChatImageURL{URL: "file:///etc/passwd"}},
			}}}},
			then: `{"messages":[{"role":"user","content":[
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},
				{"type":"image","source":{"type":"url","url":"https://x/y.png"}}
			]}]}`,
		},
		{
			name: "GIVEN a cache breakpoint on a part of the conversation THEN it is kept",
			given: proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{{Role: proxias.ChatMessageRoleUser, Content: proxias.ChatContent{
				{Type: proxias.ChatContentPartTypeText, Text: "a long document", CacheControl: &proxias.ChatCacheControl{Type: "ephemeral"}},
			}}}},
			then: `{"messages":[{"role":"user","content":[{"type":"text","text":"a long document","cache_control":{"type":"ephemeral"}}]}]}`,
		},
		{
			name: "GIVEN tools THEN they are Claude's, and calls are auto, unless parallel calls are refused",
			given: proxias.ChatCompletionRequest{
				Messages:          userSays("hello").Messages,
				Tools:             []proxias.ChatTool{{Type: "function", Function: proxias.ChatFunction{Name: "restart", Description: "Restarts", Parameters: map[string]any{"type": "object", "properties": map[string]any{"service": map[string]any{"type": "string"}}, "required": []any{"service"}}}}},
				ToolChoice:        json.RawMessage(`"required"`),
				ParallelToolCalls: &no,
			},
			then: `{"tools":[{"name":"restart","description":"Restarts","input_schema":{"type":"object","properties":{"service":{"type":"string"}},"required":["service"]}}],"tool_choice":{"type":"auto","disable_parallel_tool_use":true}}`,
		},
		{
			name: "GIVEN tool_choice none THEN no tool is used",
			given: proxias.ChatCompletionRequest{
				Messages:   userSays("hello").Messages,
				Tools:      []proxias.ChatTool{{Type: "function", Function: proxias.ChatFunction{Name: "restart"}}},
				ToolChoice: json.RawMessage(`"none"`),
			},
			then: `{"tool_choice":{"type":"none"}}`,
		},
		{
			name: "GIVEN a limit of tokens and stop sequences THEN they are sent, blank ones apart",
			given: proxias.ChatCompletionRequest{
				Messages:            userSays("hello").Messages,
				MaxCompletionTokens: 100,
				Stop:                json.RawMessage(`["END"," "]`),
			},
			then: `{"max_tokens":100,"stop_sequences":["END"]}`,
		},
		{
			name: "GIVEN empty messages THEN they are left out",
			given: proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{
				{Role: proxias.ChatMessageRoleUser, Content: text("hello")},
				{Role: proxias.ChatMessageRoleAssistant},
				{Role: proxias.ChatMessageRoleUser},
			}},
			then: `{"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, sent := mockAnthropic(t, 200, textMessage)
			engine := anthropic.NewReasoningEngine(client, someModel)

			_, err := engine.Reason(t.Context(), tt.given, nil)

			require.NoError(t, err)
			var want map[string]json.RawMessage
			require.NoError(t, json.Unmarshal([]byte(tt.then), &want))
			fields := make([]string, 0, len(want))
			for field := range want {
				fields = append(fields, field)
			}
			assert.JSONEq(t, tt.then, picked(t, *sent, fields...))
		})
	}
}

func TestReasoningEngineReasonReturnsAnErrorForARoleItCannotMap(t *testing.T) {
	// Given
	client, sent := mockAnthropic(t, 200, textMessage)
	engine := anthropic.NewReasoningEngine(client, someModel)
	req := proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{{Role: "critic", Content: text("hm")}}}

	// When
	_, err := engine.Reason(t.Context(), req, nil)

	// Then
	assert.ErrorContains(t, err, `message 0: unsupported role "critic"`)
	assert.Empty(t, *sent, "the API was not called")
}

func TestReasoningEngineReasonMapsTheAnswer(t *testing.T) {
	tests := []struct {
		name  string
		given string // the API's message
		then  string // the chat completion
	}{
		{
			name:  "canonical",
			given: textMessage,
			then: `{"id":"msg_1","object":"chat.completion","created":0,"model":"claude-x","choices":[{"index":0,"message":{"role":"assistant","content":"Hi there"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":15,"completion_tokens":4,"total_tokens":19,"prompt_tokens_details":{"cached_tokens":5}}}`,
		},
		{
			name:  "GIVEN a tool use THEN it is a tool call, and there is no content",
			given: `{"id":"msg_2","model":"claude-x","stop_reason":"tool_use","content":[{"type":"tool_use","id":"tu1","name":"restart","input":{"service":"api"}}],"usage":{"input_tokens":1,"output_tokens":2}}`,
			then: `{"id":"msg_2","object":"chat.completion","created":0,"model":"claude-x","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"tu1","type":"function","function":{"name":"restart","arguments":"{\"service\":\"api\"}"}}]},"finish_reason":"tool_calls"}],
				"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,"prompt_tokens_details":{"cached_tokens":0}}}`,
		},
		{
			name:  "GIVEN an answer cut at the limit THEN the finish reason is length",
			given: `{"id":"msg_3","model":"claude-x","stop_reason":"max_tokens","content":[{"type":"text","text":"Hi"}],"usage":{}}`,
			then: `{"id":"msg_3","object":"chat.completion","created":0,"model":"claude-x","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"length"}],
				"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"prompt_tokens_details":{"cached_tokens":0}}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, _ := mockAnthropic(t, 200, tt.given)
			engine := anthropic.NewReasoningEngine(client, someModel)

			got, err := engine.Reason(t.Context(), userSays("hello"), nil)

			require.NoError(t, err)
			defer func() { _ = got.Body.Close() }()
			assert.Equal(t, http.StatusOK, got.Status)
			assert.Equal(t, "application/json", got.Header.Get("Content-Type"))
			var completion proxias.ChatCompletion
			require.NoError(t, json.NewDecoder(got.Body).Decode(&completion))
			assert.NotZero(t, completion.Created)
			completion.Created = 0
			raw, err := json.Marshal(completion)
			require.NoError(t, err)
			assert.JSONEq(t, tt.then, string(raw))
		})
	}
}

func TestReasoningEngineReasonReturnsTheAPIError(t *testing.T) {
	tests := []struct {
		name  string
		given proxias.ChatCompletionRequest
	}{
		{name: "canonical", given: userSays("hello")},
		{name: "GIVEN a stream THEN the error is returned too, not sent inside the stream", given: proxias.ChatCompletionRequest{Stream: true, Messages: userSays("hello").Messages}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, _ := mockAnthropic(t, 429, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
			engine := anthropic.NewReasoningEngine(client, someModel)

			got, err := engine.Reason(t.Context(), tt.given, nil)

			var apiErr *anthropicsdk.Error
			require.ErrorAs(t, err, &apiErr)
			assert.Equal(t, 429, apiErr.StatusCode)
			assert.ErrorContains(t, err, "anthropic: ")
			assert.Nil(t, got.Body)
		})
	}
}

func TestReasoningEngineReasonStreamsTheAnswer(t *testing.T) {
	const start = `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[],"usage":{"input_tokens":10,"cache_read_input_tokens":5,"output_tokens":1}}}`
	tests := []struct {
		name  string
		given proxias.ChatCompletionRequest
		api   string
		then  []string // the data of the events of the stream, in order
	}{
		{
			name:  "canonical",
			given: proxias.ChatCompletionRequest{Stream: true, Messages: userSays("hello").Messages},
			api: sse(start,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi "}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"there"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
				`{"type":"message_stop"}`),
			then: []string{
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{"content":"Hi "},"finish_reason":null}]}`,
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{"content":"there"},"finish_reason":null}]}`,
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
				`[DONE]`,
			},
		},
		{
			name:  "GIVEN the request asks for the usage THEN a last chunk has it",
			given: proxias.ChatCompletionRequest{Stream: true, StreamOptions: proxias.ChatStreamOptions{IncludeUsage: true}, Messages: userSays("hello").Messages},
			api: sse(start,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
				`{"type":"message_stop"}`),
			then: []string{
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[],"usage":{"prompt_tokens":15,"completion_tokens":4,"total_tokens":19,"prompt_tokens_details":{"cached_tokens":5}}}`,
				`[DONE]`,
			},
		},
		{
			name:  "GIVEN a tool use THEN the call is streamed by pieces, and the finish reason is tool_calls",
			given: proxias.ChatCompletionRequest{Stream: true, Messages: userSays("hello").Messages},
			api: sse(start,
				`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"restart","input":{}}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"service\":"}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"api\"}"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}`,
				`{"type":"message_stop"}`),
			then: []string{
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tu1","type":"function","function":{"name":"restart"}}]},"finish_reason":null}]}`,
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"service\":"}}]},"finish_reason":null}]}`,
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"api\"}"}}]},"finish_reason":null}]}`,
				`{"id":"msg_1","object":"chat.completion.chunk","created":0,"model":"claude-x","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
				`[DONE]`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, _ := mockAnthropic(t, 200, tt.api)
			engine := anthropic.NewReasoningEngine(client, someModel)

			got, err := engine.Reason(t.Context(), tt.given, nil)

			require.NoError(t, err)
			defer func() { _ = got.Body.Close() }()
			assert.Equal(t, http.StatusOK, got.Status)
			assert.Equal(t, "text/event-stream", got.Header.Get("Content-Type"))
			raw, err := io.ReadAll(got.Body)
			require.NoError(t, err)

			events := strings.Split(strings.TrimSpace(string(raw)), "\n\n")
			require.Len(t, events, len(tt.then))
			for i, event := range events {
				data := strings.TrimPrefix(event, "data: ")
				if tt.then[i] == "[DONE]" {
					assert.Equal(t, "[DONE]", data)
					continue
				}
				var chunk proxias.ChatCompletionChunk
				require.NoError(t, json.Unmarshal([]byte(data), &chunk))
				chunk.Created = 0
				normalized, err := json.Marshal(chunk)
				require.NoError(t, err)
				assert.JSONEq(t, tt.then[i], string(normalized))
			}
		})
	}
}

func TestReasoningEngineReasonAsksTheAPIToStreamOnlyWhenTheHarnessDoes(t *testing.T) {
	tests := []struct {
		name  string
		given bool
		then  bool
	}{
		{name: "GIVEN a request that does not stream THEN the API is not streamed", given: false, then: false},
		{name: "GIVEN a request that streams THEN the API is streamed", given: true, then: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := textMessage
			if tt.given {
				api = sse(`{"type":"message_start","message":{"id":"m","model":"x","content":[],"usage":{}}}`, `{"type":"message_stop"}`)
			}
			client, sent := mockAnthropic(t, 200, api)
			engine := anthropic.NewReasoningEngine(client, someModel)

			got, err := engine.Reason(t.Context(), proxias.ChatCompletionRequest{Stream: tt.given, Messages: userSays("hello").Messages}, nil)

			require.NoError(t, err)
			_ = got.Body.Close()
			if tt.then {
				assert.Contains(t, *sent, `"stream":true`)
			} else {
				assert.NotContains(t, *sent, `"stream"`)
			}
		})
	}
}
