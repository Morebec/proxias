package openaihttp_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/transports/openaihttp"
)

func TestHandlerModels(t *testing.T) {
	tests := []struct {
		name   string
		agents []proxias.Agent
		then   string
	}{
		{
			name:   "GIVEN agents THEN they are the models",
			agents: []proxias.Agent{{ID: "ada"}, {ID: "max"}},
			then: `{"object":"list","data":[
				{"id":"ada","object":"model","created":0,"owned_by":"proxias"},
				{"id":"max","object":"model","created":0,"owned_by":"proxias"}]}`,
		},
		{
			name: "GIVEN no agent THEN there is no model",
			then: `{"object":"list","data":[]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := openaihttp.NewHandler(
				proxias.StrategyFunc(
					func(context.Context, proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
						return func(func(proxias.StreamUpdate) bool) {}
					},
				),
				tt.agents,
			)

			w := httptest.NewRecorder()
			handler.ServeHTTP(w, httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/v1/models", nil))

			assert.Equal(t, http.StatusOK, w.Code)
			assert.JSONEq(t, tt.then, w.Body.String())
		})
	}
}

func TestHandlerChatCompletions(t *testing.T) {
	tests := []struct {
		name   string
		turn   []proxias.StreamUpdate
		err    error
		when   string
		code   int
		then   string
		stream []string
		told   []string
	}{
		{
			name: "WHEN messages THEN they are the session, and the reply a chat.completion",
			turn: []proxias.StreamUpdate{
				{
					Events: []proxias.Event{
						proxias.NewEvent(
							proxias.EventTypeMessage,
							proxias.NewAgentSource("ada"),
							[]byte("Hi John."),
						),
					},
					StopReason: proxias.StopReasonEndTurn,
				},
			},
			when: `{"model":"ada","messages":[
				{"role":"system","content":"Be brief."},
				{"role":"user","content":"hello"},
				{"role":"assistant","content":"Hello!"},
				{"role":"user","content":[{"type":"text","text":"I am John."}]},
				{"role":"tool","content":"ignored","tool_call_id":"call_1"}]}`,
			code: http.StatusOK,
			then: `{"object":"chat.completion","model":"ada","choices":[{"index":0,"message":{"role":"assistant","content":"Hi John."},"finish_reason":"stop"}]}`,
			told: []string{
				"system:instructions Be brief.",
				"user:user hello",
				"agent:ada Hello!",
				"user:user I am John.",
			},
		},
		{
			name: "WHEN stream THEN the reply's chunks are SSE chunks, ending with [DONE], a thought's text left out",
			turn: []proxias.StreamUpdate{
				{
					Chunk: &proxias.StreamChunk{
						EventID: "t",
						Data:    []byte("Thinking."),
					},
				},
				{
					Events: []proxias.Event{
						{
							ID:      "t",
							Type:    proxias.EventTypeThought,
							Source:  proxias.NewAgentSource("ada"),
							Payload: []byte("Thinking."),
							Date:    time.Now(),
						},
					},
				},
				{
					Chunk: &proxias.StreamChunk{
						EventID: "r",
						Data:    []byte("Hi "),
					},
				},
				{
					Chunk: &proxias.StreamChunk{
						EventID: "r",
						Data:    []byte("John."),
					},
				},
				{
					Events: []proxias.Event{
						{
							ID:      "r",
							Type:    proxias.EventTypeMessage,
							Source:  proxias.NewAgentSource("ada"),
							Payload: []byte("Hi John."),
							Date:    time.Now(),
						},
					},
					StopReason: proxias.StopReasonEndTurn,
				},
			},
			when: `{"model":"ada","stream":true,"messages":[{"role":"user","content":"hello"}]}`,
			code: http.StatusOK,
			stream: []string{
				`{"content":"","role":"assistant"}`,
				`{"content":"Hi "}`,
				`{"content":"John."}`,
				`{}`,
				"[DONE]",
			},
			told: []string{"user:user hello"},
		},
		{
			name: "WHEN the model is not an agent THEN it is not found",
			when: `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`,
			code: http.StatusNotFound,
			then: `{"error":{"message":"the model \"gpt-4\" does not exist","type":"model_not_found","param":null,"code":null}}`,
		},
		{
			name: "WHEN no message THEN it is a bad request",
			when: `{"model":"ada","messages":[]}`,
			code: http.StatusBadRequest,
			then: `{"error":{"message":"messages is empty","type":"invalid_request_error","param":null,"code":null}}`,
		},
		{
			name: "WHEN not JSON THEN it is a bad request",
			when: `hello`,
			code: http.StatusBadRequest,
			then: `{"error":{"message":"the body is not a chat completion request","type":"invalid_request_error","param":null,"code":null}}`,
		},
		{
			name: "WHEN the agent fails THEN it is a server error",
			err:  errors.New("down"),
			when: `{"model":"ada","messages":[{"role":"user","content":"hello"}]}`,
			code: http.StatusInternalServerError,
			then: `{"error":{"message":"down","type":"server_error","param":null,"code":null}}`,
			told: []string{"user:user hello"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var told []string
			strategy := proxias.StrategyFunc(
				func(_ context.Context, inv proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
					for _, e := range inv.Session.Events() {
						told = append(told, e.Source.String()+" "+e.Text())
					}
					return func(yield func(proxias.StreamUpdate) bool) {
						for _, d := range tt.turn {
							if !yield(d) {
								return
							}
						}
						if tt.err != nil {
							yield(proxias.StreamUpdate{Err: tt.err})
						}
					}
				},
			)
			handler := openaihttp.NewHandler(
				strategy,
				[]proxias.Agent{{ID: "ada"}},
			)

			w := httptest.NewRecorder()
			handler.ServeHTTP(
				w,
				httptest.NewRequestWithContext(
					t.Context(),
					http.MethodPost,
					"/v1/chat/completions",
					strings.NewReader(tt.when),
				),
			)

			assert.Equal(t, tt.code, w.Code)
			assert.Equal(t, tt.told, told)
			if tt.stream == nil {
				var body map[string]any
				require.NoError(t, json.Unmarshal(w.Body.Bytes(), &body))
				delete(body, "id")
				delete(body, "created")
				encoded, _ := json.Marshal(body)
				assert.JSONEq(t, tt.then, string(encoded))
				return
			}
			assert.Equal(t, "text/event-stream", w.Header().Get("Content-Type"))
			var deltas []string
			for _, line := range strings.Split(strings.TrimSpace(w.Body.String()), "\n\n") {
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					deltas = append(deltas, data)
					continue
				}
				var chunk struct {
					Object  string `json:"object"`
					Choices []struct {
						Delta map[string]any `json:"delta"`
					} `json:"choices"`
				}
				require.NoError(t, json.Unmarshal([]byte(data), &chunk))
				assert.Equal(t, "chat.completion.chunk", chunk.Object)
				delta, _ := json.Marshal(chunk.Choices[0].Delta)
				deltas = append(deltas, string(delta))
			}
			assert.Equal(t, tt.stream, deltas)
		})
	}
}
