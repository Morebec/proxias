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

func TestLLMComplete(t *testing.T) {
	tests := []struct {
		name string
		// answer is the message of the completion the API answers, finishReason
		// its finish_reason ("stop" when empty).
		answer, finishReason string
		when                 proxias.LLMRequest
		then                 []proxias.LLMChunk
		messages             string
		tools                string
	}{
		{
			name:   "GIVEN the server sends the model's reasoning THEN it is its thinking, before its answer",
			answer: `{"role":"assistant","content":"Hi.","reasoning_content":"The user greets me."}`,
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{Role: proxias.LLMMessageRoleUser, Parts: []proxias.LLMPart{{Text: "hello"}}},
			}},
			then: []proxias.LLMChunk{
				{Thinking: "The user greets me."},
				{Text: "Hi."},
				{StopReason: proxias.StopReasonEndTurn},
			},
			messages: `[{"role":"user","content":"hello"}]`,
		},
		{
			name:   "GIVEN files THEN each is named, then given as the model reads it: an image, a PDF file, text, or only its name",
			answer: `{"role":"assistant","content":"ok"}`,
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{Role: proxias.LLMMessageRoleUser, Parts: []proxias.LLMPart{
					{Text: "look"},
					{File: &proxias.File{Name: "photo.png", MediaType: "image/png", Content: []byte("png")}},
					{File: &proxias.File{Name: "quote.pdf", MediaType: "application/pdf", Content: []byte("pdf")}},
					{File: &proxias.File{Name: "notes.csv", MediaType: "text/csv", Content: []byte("a,b")}},
					{File: &proxias.File{Name: "plan.docx", MediaType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", Content: []byte("docx")}},
				}},
			}},
			then: []proxias.LLMChunk{{Text: "ok"}, {StopReason: proxias.StopReasonEndTurn}},
			messages: `[
				{"role":"user","content":[
					{"type":"text","text":"look"},
					{"type":"text","text":"[file photo.png, image/png]"},
					{"type":"image_url","image_url":{"url":"data:image/png;base64,cG5n"}},
					{"type":"text","text":"[file quote.pdf, application/pdf]"},
					{"type":"file","file":{"file_data":"data:application/pdf;base64,cGRm","filename":"quote.pdf"}},
					{"type":"text","text":"[file notes.csv, text/csv]\na,b"},
					{"type":"text","text":"[file plan.docx, application/vnd.openxmlformats-officedocument.wordprocessingml.document, which you cannot open]"}
				]}
			]`,
		},
		{
			name:   "GIVEN the assistant's thinking THEN its own is given back as reasoning_content, another vendor's, signed, left out",
			answer: `{"role":"assistant","content":"ok"}`,
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{Role: proxias.LLMMessageRoleUser, Parts: []proxias.LLMPart{{Text: "hello"}}},
				{Role: proxias.LLMMessageRoleAssistant, Parts: []proxias.LLMPart{
					{Thinking: "The user greets me."},
					{Thinking: "Claude's thinking.", Signature: "sig1"},
					{RedactedThinking: "secret"},
					{Text: "Hi."},
				}},
				{Role: proxias.LLMMessageRoleUser, Parts: []proxias.LLMPart{{Text: "again"}}},
			}},
			then: []proxias.LLMChunk{{Text: "ok"}, {StopReason: proxias.StopReasonEndTurn}},
			messages: `[
				{"role":"user","content":"hello"},
				{"role":"assistant","content":"Hi.","reasoning_content":"The user greets me."},
				{"role":"user","content":"again"}
			]`,
		},
		{
			name:   "GIVEN a text answer THEN it is a text chunk, and the system prompt system messages",
			answer: `{"role":"assistant","content":"Hello John."}`,
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{
					Role:  proxias.LLMMessageRoleSystem,
					Parts: []proxias.LLMPart{{Text: "You are Ada."}},
				},
				{
					Role:  proxias.LLMMessageRoleSystem,
					Parts: []proxias.LLMPart{{Text: "Be brief."}},
				},
				{
					Role:  proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{{Text: "hi"}},
				},
			}},
			then: []proxias.LLMChunk{
				{Text: "Hello John."},
				{StopReason: proxias.StopReasonEndTurn},
			},
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
			when: proxias.LLMRequest{
				Tools: []proxias.ActionDescriptor{
					{
						ID:          "send_email",
						Description: "Sends an email.",
						InputSchema: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"to": map[string]any{"type": "string"},
							},
						},
					},
					{ID: "get_time", Description: "Tells the time."},
				},
				Messages: []proxias.LLMMessage{
					{
						Role:  proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{{Text: "email joe"}},
					},
				},
			},
			then: []proxias.LLMChunk{
				{Text: "An email."},
				{
					ToolCall: &proxias.LLMToolCall{
						ID:    "call_1",
						Name:  "send_email",
						Input: map[string]any{"to": "joe"},
					},
				},
				{
					ToolCall: &proxias.LLMToolCall{
						ID:   "call_2",
						Name: "get_time",
					},
				},
				{StopReason: proxias.StopReasonEndTurn},
			},
			messages: `[{"role":"user","content":"email joe"}]`,
			tools: `[
				{"type":"function","function":{"name":"send_email","description":"Sends an email.","parameters":{"type":"object","properties":{"to":{"type":"string"}}}}},
				{"type":"function","function":{"name":"get_time","description":"Tells the time.","parameters":{"type":"object","properties":{}}}}
			]`,
		},
		{
			name:   "GIVEN a system message after the first turn THEN it is a system message, in place",
			answer: `{"role":"assistant","content":"ok"}`,
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{
					Role:  proxias.LLMMessageRoleSystem,
					Parts: []proxias.LLMPart{{Text: "Be brief."}},
				},
				{
					Role:  proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{{Text: "hi"}},
				},
				{
					Role:  proxias.LLMMessageRoleSystem,
					Parts: []proxias.LLMPart{{Text: "It is noon."}},
				},
				{
					Role:  proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{{Text: "what time is it?"}},
				},
			}},
			then: []proxias.LLMChunk{
				{Text: "ok"},
				{StopReason: proxias.StopReasonEndTurn},
			},
			messages: `[
				{"role":"system","content":"Be brief."},
				{"role":"user","content":"hi"},
				{"role":"system","content":"It is noon."},
				{"role":"user","content":"what time is it?"}
			]`,
		},
		{
			name:   "GIVEN empty text parts THEN they are left out, and a message left empty with them",
			answer: `{"role":"assistant","content":"ok"}`,
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{
					Role:  proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{{Text: "hi"}, {Text: ""}},
				},
				{
					Role:  proxias.LLMMessageRoleAssistant,
					Parts: []proxias.LLMPart{{Text: ""}},
				},
				{
					Role:  proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{{Text: "still there?"}},
				},
			}},
			then: []proxias.LLMChunk{
				{Text: "ok"},
				{StopReason: proxias.StopReasonEndTurn},
			},
			messages: `[{"role":"user","content":"hi"},{"role":"user","content":"still there?"}]`,
		},
		{
			name:   "GIVEN tool calls and their results THEN they are the assistant's tool calls and tool messages, thinking left out",
			answer: `{"role":"assistant","content":"Sent."}`,
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{
					Role:  proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{{Text: "email joe"}},
				},
				{
					Role: proxias.LLMMessageRoleAssistant,
					Parts: []proxias.LLMPart{
						{Thinking: "An email.", Signature: "sig"},
						{Text: "Sending."},
						{
							ToolCall: &proxias.LLMToolCall{
								ID:    "call_1",
								Name:  "send_email",
								Input: map[string]any{"to": "joe"},
							},
						},
					},
				},
				{Role: proxias.LLMMessageRoleUser, Parts: []proxias.LLMPart{
					{
						ToolResult: &proxias.LLMToolResult{
							CallID:  "call_1",
							Content: "sent",
						},
					},
					{Text: "[probe.disk from system:db-1] 93%"},
				}},
			}},
			then: []proxias.LLMChunk{
				{Text: "Sent."},
				{StopReason: proxias.StopReasonEndTurn},
			},
			messages: `[
				{"role":"user","content":"email joe"},
				{"role":"assistant","content":"Sending.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"send_email","arguments":"{\"to\":\"joe\"}"}}]},
				{"role":"tool","tool_call_id":"call_1","content":"sent"},
				{"role":"user","content":"[probe.disk from system:db-1] 93%"}
			]`,
		}, {
			name:         "GIVEN the model stops at the limit of tokens THEN the answer so far, and the length",
			answer:       `{"role":"assistant","content":"Once upon"}`,
			finishReason: "length",
			when: proxias.LLMRequest{
				Messages: []proxias.LLMMessage{
					{
						Role:  proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{{Text: "tell me a story"}},
					},
				},
			},
			then: []proxias.LLMChunk{
				{Text: "Once upon"},
				{StopReason: proxias.StopReasonLength},
			},
			messages: `[{"role":"user","content":"tell me a story"}]`,
		},
		{
			name:         "GIVEN the provider filters the answer THEN the content filter",
			answer:       `{"role":"assistant","content":""}`,
			finishReason: "content_filter",
			when: proxias.LLMRequest{
				Messages: []proxias.LLMMessage{
					{
						Role:  proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{{Text: "something harmful"}},
					},
				},
			},
			then: []proxias.LLMChunk{
				{StopReason: proxias.StopReasonContentFilter},
			},
			messages: `[{"role":"user","content":"something harmful"}]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var request map[string]any
			finishReason := tt.finishReason
			if finishReason == "" {
				finishReason = "stop"
			}
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					_ = json.NewDecoder(r.Body).Decode(&request)
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(
						[]byte(
							`{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"finish_reason":"` + finishReason + `","message":` + tt.answer + `}]}`,
						),
					)
				}),
			)
			defer server.Close()
			client := openaisdk.NewClient(
				option.WithBaseURL(server.URL),
				option.WithAPIKey("test"),
				option.WithMaxRetries(0),
			)

			var chunks []proxias.LLMChunk
			llm := openai.NewLLM(client, "m")
			llm.Buffered = true
			for c, err := range llm.Complete(context.Background(), tt.when) {
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
		{
			name:   "GIVEN a bad key THEN it is the SDK's error",
			status: http.StatusUnauthorized,
			body:   `{"error":{"message":"bad key","type":"invalid_request_error"}}`,
		},
		{
			name:   "GIVEN a rate limit THEN it is the SDK's error",
			status: http.StatusTooManyRequests,
			body:   `{"error":{"message":"slow down","type":"rate_limit_error"}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(tt.status)
					_, _ = w.Write([]byte(tt.body))
				}),
			)
			defer server.Close()
			client := openaisdk.NewClient(
				option.WithBaseURL(server.URL),
				option.WithAPIKey("test"),
				option.WithMaxRetries(0),
			)

			var err error
			llm := openai.NewLLM(client, "m")
			llm.Buffered = true
			for _, e := range llm.Complete(context.Background(), proxias.LLMRequest{}) {
				err = e
			}

			var apiErr *openaisdk.Error
			require.ErrorAs(t, err, &apiErr)
			assert.Equal(t, tt.status, apiErr.StatusCode)
		})
	}
}

func TestLLMCompleteStreaming(t *testing.T) {
	tests := []struct {
		name   string
		chunks string
		then   []proxias.LLMChunk
	}{
		{
			name: "Canonical case: the text is handed out piece by piece, then why the model stopped",
			chunks: `data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello "}}]}` + "\n\n" +
				`data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"Betty"},"finish_reason":"stop"}]}` + "\n\n" +
				"data: [DONE]\n\n",
			then: []proxias.LLMChunk{
				{
					Text: "Hello ",
				},
				{
					Text: "Betty",
				},
				{
					StopReason: proxias.StopReasonEndTurn,
				},
			},
		},
		{
			name: "GIVEN a tool call, THEN it is handed out whole, once the answer is, and the model did not stop",
			chunks: `data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"t1","type":"function","function":{"name":"clock","arguments":"{\"zone\":"}}]}}]}` + "\n\n" +
				`data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"UTC\"}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
				"data: [DONE]\n\n",
			then: []proxias.LLMChunk{
				{
					ToolCall: &proxias.LLMToolCall{
						ID:   "t1",
						Name: "clock",
						Input: map[string]any{
							"zone": "UTC",
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var request map[string]any
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					_ = json.NewDecoder(r.Body).Decode(&request)
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte(tt.chunks))
				}),
			)
			defer server.Close()
			client := openaisdk.NewClient(
				option.WithBaseURL(server.URL),
				option.WithAPIKey("test"),
				option.WithMaxRetries(0),
			)
			llm := openai.NewLLM(client, "m")

			var chunks []proxias.LLMChunk
			for c, err := range llm.Complete(context.Background(), proxias.LLMRequest{}) {
				require.NoError(t, err)
				chunks = append(chunks, c)
			}

			assert.Equal(t, tt.then, chunks)
			assert.Equal(t, true, request["stream"])
			assert.Equal(t, map[string]any{"include_usage": true}, request["stream_options"])
		})
	}
}

func TestLLMCompleteEffort(t *testing.T) {
	tests := []struct {
		name  string
		given proxias.Effort
		when  proxias.Effort
		then  any
	}{
		{
			name:  "canonical",
			given: proxias.EffortLow,
			when:  "",
			then:  "low",
		},
		{
			name:  "GIVEN an effort WHEN the request names another THEN the request's applies",
			given: proxias.EffortLow,
			when:  proxias.EffortHigh,
			then:  "high",
		},
		{
			name:  "GIVEN no effort WHEN the request names none THEN the model's default applies",
			given: "",
			when:  "",
			then:  nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var request map[string]any
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					_ = json.NewDecoder(r.Body).Decode(&request)
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(
						[]byte(
							`{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Hi."}}]}`,
						),
					)
				}),
			)
			defer server.Close()
			client := openaisdk.NewClient(
				option.WithBaseURL(server.URL),
				option.WithAPIKey("test"),
				option.WithMaxRetries(0),
			)
			llm := openai.NewLLM(client, "m")
			llm.Buffered = true
			llm.Effort = tt.given

			for _, err := range llm.Complete(context.Background(), proxias.LLMRequest{
				Messages: []proxias.LLMMessage{
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "Hi"},
						},
					},
				},
				Effort: tt.when,
			}) {
				require.NoError(t, err)
			}

			assert.Equal(t, tt.then, request["reasoning_effort"])
		})
	}
}

func TestLLMCompleteModel(t *testing.T) {
	tests := []struct {
		name string
		when string
		then any
	}{
		{
			name: "canonical",
			when: "",
			then: "gpt-5",
		},
		{
			name: "WHEN the request names a model THEN it is asked",
			when: "gpt-5-mini",
			then: "gpt-5-mini",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var request map[string]any
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					_ = json.NewDecoder(r.Body).Decode(&request)
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(
						[]byte(
							`{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Hi."}}]}`,
						),
					)
				}),
			)
			defer server.Close()
			client := openaisdk.NewClient(
				option.WithBaseURL(server.URL),
				option.WithAPIKey("test"),
				option.WithMaxRetries(0),
			)
			llm := openai.NewLLM(client, "gpt-5")
			llm.Buffered = true

			for _, err := range llm.Complete(context.Background(), proxias.LLMRequest{
				Messages: []proxias.LLMMessage{
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "Hi"},
						},
					},
				},
				Model: tt.when,
			}) {
				require.NoError(t, err)
			}

			assert.Equal(t, tt.then, request["model"])
		})
	}
}
