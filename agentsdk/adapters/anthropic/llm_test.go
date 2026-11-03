package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
	"github.com/morebec/proxias/agentsdk/adapters/anthropic"
)

// mockClaude is a Messages API that answers status and body, and records the
// request it was sent.
func mockClaude(t *testing.T, status int, body string, request *map[string]any) anthropicsdk.Client {
	t.Helper()
	server := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(request)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			_, _ = w.Write([]byte(body))
		}),
	)
	t.Cleanup(server.Close)
	return anthropicsdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test"), option.WithMaxRetries(0))
}

func TestLLMComplete(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		answer   string
		when     agentsdk.LLMRequest
		then     []agentsdk.LLMChunk
		messages string
		system   string
		tools    string
		// thinking configures the LLM's thinking; requested is the thinking the
		// request sends, "" for none.
		thinking  anthropicsdk.ThinkingConfigParamUnion
		requested string
	}{
		{
			name:   "GIVEN another vendor's thinking, unsigned THEN it is left out: Claude takes only its own, signed",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"ok"}]}`,
			when: agentsdk.LLMRequest{Messages: []agentsdk.LLMMessage{
				{Role: agentsdk.LLMMessageRoleUser, Parts: []agentsdk.LLMPart{{Text: "hello"}}},
				{Role: agentsdk.LLMMessageRoleAssistant, Parts: []agentsdk.LLMPart{{Thinking: "The user greets me."}, {Text: "Hi."}}},
				{Role: agentsdk.LLMMessageRoleUser, Parts: []agentsdk.LLMPart{{Text: "again"}}},
			}},
			then: []agentsdk.LLMChunk{{Text: "ok"}, {StopReason: agentsdk.StopReasonEndTurn}},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"hello"}]},
				{"role":"assistant","content":[{"type":"text","text":"Hi."}]},
				{"role":"user","content":[{"type":"text","text":"again"}]}
			]`,
		},
		{
			name:   "GIVEN thinking configured THEN it is requested, and the thinking blocks are handed out first",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"tool_use","usage":{},
				"content":[{"type":"thinking","thinking":"The user wants the time.","signature":"sig1"},{"type":"redacted_thinking","data":"secret"},
				{"type":"tool_use","id":"toolu_1","name":"get_time","input":{}}]}`,
			when: agentsdk.LLMRequest{Messages: []agentsdk.LLMMessage{
				{
					Role:  agentsdk.LLMMessageRoleUser,
					Parts: []agentsdk.LLMPart{{Text: "what time is it?"}},
				},
			}},
			thinking: anthropicsdk.ThinkingConfigParamUnion{
				OfAdaptive: &anthropicsdk.ThinkingConfigAdaptiveParam{
					Display: anthropicsdk.ThinkingConfigAdaptiveDisplaySummarized,
				},
			},
			requested: `{"type":"adaptive","display":"summarized"}`,
			then: []agentsdk.LLMChunk{
				{Thinking: "The user wants the time.", Signature: "sig1"},
				{RedactedThinking: "secret"},
				{
					ToolCall: &agentsdk.LLMToolCall{
						ID:    "toolu_1",
						Name:  "get_time",
						Input: map[string]any{},
					},
				},
			},
			messages: `[{"role":"user","content":[{"type":"text","text":"what time is it?"}]}]`,
		},
		{
			name:   "GIVEN the thinking of an earlier answer THEN it is given back signed, as it came, before its tool call",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"It is noon."}]}`,
			when: agentsdk.LLMRequest{
				Tools: []agentsdk.ActionDescriptor{{ID: "get_time"}},
				Messages: []agentsdk.LLMMessage{
					{
						Role:  agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{{Text: "what time is it?"}},
					},
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{
								Thinking:  "The user wants the time.",
								Signature: "sig1",
							},
							{RedactedThinking: "secret"},
							{
								ToolCall: &agentsdk.LLMToolCall{
									ID:   "toolu_1",
									Name: "get_time",
								},
							},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{
							{
								ToolResult: &agentsdk.LLMToolResult{
									CallID:  "toolu_1",
									Content: "noon",
								},
							},
						},
					},
				},
			},
			then: []agentsdk.LLMChunk{
				{Text: "It is noon."},
				{StopReason: agentsdk.StopReasonEndTurn},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"what time is it?"}]},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"The user wants the time.","signature":"sig1"},
					{"type":"redacted_thinking","data":"secret"},
					{"type":"tool_use","id":"toolu_1","name":"get_time","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"noon"}],"is_error":false}]}
			]`,
		},
		{
			name:   "GIVEN a conversation with a tool call and its result THEN they are Claude's blocks, and the answer its chunks",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"tool_use","usage":{},
				"content":[{"type":"text","text":"An email."},{"type":"tool_use","id":"toolu_1","name":"send_email","input":{"to":"joe"}}]}`,
			when: agentsdk.LLMRequest{

				Tools: []agentsdk.ActionDescriptor{
					{
						ID:          "send_email",
						Description: "Sends an email.",
						InputSchema: map[string]any{
							"type": "object", "properties": map[string]any{"to": map[string]any{"type": "string"}}, "required": []string{"to"},
						},
					},
				},
				Messages: []agentsdk.LLMMessage{
					{
						Role:  agentsdk.LLMMessageRoleSystem,
						Parts: []agentsdk.LLMPart{{Text: "You are Ada."}},
					},
					{
						Role:  agentsdk.LLMMessageRoleSystem,
						Parts: []agentsdk.LLMPart{{Text: "Be brief."}},
					},
					{
						Role:  agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{{Text: "email joe"}},
					},
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{
								ToolCall: &agentsdk.LLMToolCall{
									ID:    "toolu_0",
									Name:  "send_email",
									Input: map[string]any{"to": "joe"},
								},
							},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleTool,
						Parts: []agentsdk.LLMPart{
							{
								ToolResult: &agentsdk.LLMToolResult{
									CallID:  "toolu_0",
									Content: "sent",
								},
							},
						},
					},
				},
			},
			then: []agentsdk.LLMChunk{
				{Text: "An email."},
				{
					ToolCall: &agentsdk.LLMToolCall{
						ID:    "toolu_1",
						Name:  "send_email",
						Input: map[string]any{"to": "joe"},
					},
				},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"email joe"}]},
				{"role":"assistant","content":[{"type":"tool_use","id":"toolu_0","name":"send_email","input":{"to":"joe"}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_0","content":[{"type":"text","text":"sent"}],"is_error":false}]}
			]`,
			system: `[{"type":"text","text":"You are Ada."},{"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral"}}]`,
			tools:  `[{"name":"send_email","description":"Sends an email.","input_schema":{"type":"object","properties":{"to":{"type":"string"}},"required":["to"]}}]`,
		},
		{
			name:   "GIVEN tool calls and their results, and no tools THEN they are left out, and a message left empty with them",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"Sent."}]}`,
			when: agentsdk.LLMRequest{Messages: []agentsdk.LLMMessage{
				{
					Role:  agentsdk.LLMMessageRoleUser,
					Parts: []agentsdk.LLMPart{{Text: "email joe"}},
				},
				{
					Role: agentsdk.LLMMessageRoleAssistant,
					Parts: []agentsdk.LLMPart{
						{Text: "An email."},
						{
							ToolCall: &agentsdk.LLMToolCall{
								ID:   "toolu_0",
								Name: "send_email",
							},
						},
					},
				},
				{
					Role: agentsdk.LLMMessageRoleUser,
					Parts: []agentsdk.LLMPart{
						{
							ToolResult: &agentsdk.LLMToolResult{
								CallID:  "toolu_0",
								Content: "sent",
							},
						},
					},
				},
				{
					Role:  agentsdk.LLMMessageRoleUser,
					Parts: []agentsdk.LLMPart{{Text: "thanks"}},
				},
			}},
			then: []agentsdk.LLMChunk{
				{Text: "Sent."},
				{StopReason: agentsdk.StopReasonEndTurn},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"email joe"}]},
				{"role":"assistant","content":[{"type":"text","text":"An email."}]},
				{"role":"user","content":[{"type":"text","text":"thanks"}]}
			]`,
		},
		{
			name:   "GIVEN thinking, signed or redacted, and no tools THEN it is left out, and a message left empty with it",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"Hi."}]}`,
			when: agentsdk.LLMRequest{
				Messages: []agentsdk.LLMMessage{
					{
						Role: agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{
							{Text: "hello"},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{
								Thinking:  "The user greets me.",
								Signature: "sig",
							},
							{
								RedactedThinking: "redacted",
							},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{Text: "Hi."},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{
							{Text: "thanks"},
						},
					},
				},
			},
			then: []agentsdk.LLMChunk{
				{Text: "Hi."},
				{StopReason: agentsdk.StopReasonEndTurn},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"hello"}]},
				{"role":"assistant","content":[{"type":"text","text":"Hi."}]},
				{"role":"user","content":[{"type":"text","text":"thanks"}]}
			]`,
		},
		{
			name:   "GIVEN thinking of an earlier turn, and of the turn its results answer THEN only the latter is given back",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"It is noon."}]}`,
			when: agentsdk.LLMRequest{
				Tools: []agentsdk.ActionDescriptor{
					{ID: "get_time"},
				},
				Messages: []agentsdk.LLMMessage{
					{
						Role: agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{
							{Text: "hello"},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{
								Thinking:  "The user greets me.",
								Signature: "sig0",
							},
							{RedactedThinking: "old"},
							{Text: "Hi."},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{
							{Text: "what time is it?"},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{
								Thinking:  "The user wants the time.",
								Signature: "sig1",
							},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{
								ToolCall: &agentsdk.LLMToolCall{
									ID:   "toolu_1",
									Name: "get_time",
								},
							},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleTool,
						Parts: []agentsdk.LLMPart{
							{
								ToolResult: &agentsdk.LLMToolResult{
									CallID:  "toolu_1",
									Content: "noon",
								},
							},
						},
					},
				},
			},
			then: []agentsdk.LLMChunk{
				{Text: "It is noon."},
				{StopReason: agentsdk.StopReasonEndTurn},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"hello"}]},
				{"role":"assistant","content":[{"type":"text","text":"Hi."}]},
				{"role":"user","content":[{"type":"text","text":"what time is it?"}]},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"The user wants the time.","signature":"sig1"},
					{"type":"tool_use","id":"toolu_1","name":"get_time","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"noon"}],"is_error":false}]}
			]`,
		},
		{
			name:   "GIVEN thinking, and a conversation that does not end on tool results THEN it is left out",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"You're welcome."}]}`,
			when: agentsdk.LLMRequest{
				Tools: []agentsdk.ActionDescriptor{
					{ID: "get_time"},
				},
				Messages: []agentsdk.LLMMessage{
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{
								Thinking:  "The user greets me.",
								Signature: "sig0",
							},
							{Text: "It is noon."},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{
							{Text: "thanks"},
						},
					},
				},
			},
			then: []agentsdk.LLMChunk{
				{Text: "You're welcome."},
				{StopReason: agentsdk.StopReasonEndTurn},
			},
			messages: `[
				{"role":"assistant","content":[{"type":"text","text":"It is noon."}]},
				{"role":"user","content":[{"type":"text","text":"thanks"}]}
			]`,
		},
		{
			name:   "GIVEN a system message after the first turn THEN it is the user's, tagged, in place",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"ok"}]}`,
			when: agentsdk.LLMRequest{Messages: []agentsdk.LLMMessage{
				{
					Role:  agentsdk.LLMMessageRoleSystem,
					Parts: []agentsdk.LLMPart{{Text: "Be brief."}},
				},
				{
					Role:  agentsdk.LLMMessageRoleUser,
					Parts: []agentsdk.LLMPart{{Text: "hi"}},
				},
				{
					Role:  agentsdk.LLMMessageRoleAssistant,
					Parts: []agentsdk.LLMPart{{Text: "Hello."}},
				},
				{
					Role:  agentsdk.LLMMessageRoleSystem,
					Parts: []agentsdk.LLMPart{{Text: "It is noon."}},
				},
				{
					Role:  agentsdk.LLMMessageRoleUser,
					Parts: []agentsdk.LLMPart{{Text: "what time is it?"}},
				},
			}},
			then: []agentsdk.LLMChunk{
				{Text: "ok"},
				{StopReason: agentsdk.StopReasonEndTurn},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"hi"}]},
				{"role":"assistant","content":[{"type":"text","text":"Hello."}]},
				{"role":"user","content":[{"type":"text","text":"[system] It is noon."},{"type":"text","text":"what time is it?"}]}
			]`,
			system: `[{"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral"}}]`,
		},
		{
			name:   "GIVEN empty text parts THEN they are left out, and a message left empty with them",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"ok"}]}`,
			when: agentsdk.LLMRequest{Messages: []agentsdk.LLMMessage{
				{
					Role:  agentsdk.LLMMessageRoleUser,
					Parts: []agentsdk.LLMPart{{Text: "hi"}, {Text: ""}},
				},
				{
					Role:  agentsdk.LLMMessageRoleAssistant,
					Parts: []agentsdk.LLMPart{{Text: ""}},
				},
				{
					Role:  agentsdk.LLMMessageRoleUser,
					Parts: []agentsdk.LLMPart{{Text: "still there?"}},
				},
			}},
			then: []agentsdk.LLMChunk{
				{Text: "ok"},
				{StopReason: agentsdk.StopReasonEndTurn},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"hi"},{"type":"text","text":"still there?"}]}
			]`,
		},
		{
			name:   "GIVEN a conversation that ends on the agent THEN a user turn ends it",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"ok"}]}`,
			when: agentsdk.LLMRequest{Messages: []agentsdk.LLMMessage{
				{
					Role:  agentsdk.LLMMessageRoleAssistant,
					Parts: []agentsdk.LLMPart{{Text: "I waited."}},
				},
			}},
			then: []agentsdk.LLMChunk{
				{Text: "ok"},
				{StopReason: agentsdk.StopReasonEndTurn},
			},
			messages: `[
				{"role":"assistant","content":[{"type":"text","text":"I waited."}]},
				{"role":"user","content":[{"type":"text","text":"[no new event]"}]}
			]`,
		},
		{
			name:   "GIVEN no conversation THEN a user turn says nothing happened",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"ok"}]}`,
			then: []agentsdk.LLMChunk{
				{Text: "ok"},
				{StopReason: agentsdk.StopReasonEndTurn},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"(nothing happened yet)"}]}
			]`,
		}, {
			name:   "GIVEN Claude stops at the limit of tokens THEN the answer so far, and the length",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"max_tokens","usage":{},"content":[{"type":"text","text":"Once upon"}]}`,
			when: agentsdk.LLMRequest{
				Messages: []agentsdk.LLMMessage{
					{
						Role:  agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{{Text: "tell me a story"}},
					},
				},
			},
			then: []agentsdk.LLMChunk{
				{Text: "Once upon"},
				{StopReason: agentsdk.StopReasonLength},
			},
			messages: `[{"role":"user","content":[{"type":"text","text":"tell me a story"}]}]`,
		},
		{
			name:   "GIVEN Claude refuses THEN the content filter",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"refusal","usage":{},"content":[]}`,
			when: agentsdk.LLMRequest{
				Messages: []agentsdk.LLMMessage{
					{
						Role:  agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{{Text: "something harmful"}},
					},
				},
			},
			then: []agentsdk.LLMChunk{
				{StopReason: agentsdk.StopReasonContentFilter},
			},
			messages: `[{"role":"user","content":[{"type":"text","text":"something harmful"}]}]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var request map[string]any
			client := mockClaude(t, tt.status, tt.answer, &request)

			var chunks []agentsdk.LLMChunk
			llm := anthropic.NewLLM(client, "m")
			llm.Thinking = tt.thinking
			for c, err := range llm.Complete(context.Background(), tt.when) {
				require.NoError(t, err)
				chunks = append(chunks, c)
			}

			assert.Equal(t, tt.then, chunks)
			if tt.requested == "" {
				assert.Nil(t, request["thinking"], "the model's default thinking")
			} else {
				thinking, _ := json.Marshal(request["thinking"])
				assert.JSONEq(t, tt.requested, string(thinking))
			}
			messages, _ := json.Marshal(request["messages"])
			assert.JSONEq(t, tt.messages, string(messages))
			if tt.system != "" {
				system, _ := json.Marshal(request["system"])
				assert.JSONEq(t, tt.system, string(system), "the last system block is cached")
			}
			if tt.tools != "" {
				tools, _ := json.Marshal(request["tools"])
				assert.JSONEq(t, tt.tools, string(tools))
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
			body:   `{"type":"error","error":{"type":"authentication_error","message":"bad key"}}`,
		},
		{
			name:   "GIVEN a rate limit THEN it is the SDK's error",
			status: http.StatusTooManyRequests,
			body:   `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var request map[string]any
			client := mockClaude(t, tt.status, tt.body, &request)

			var err error
			for _, e := range anthropic.NewLLM(client, "m").Complete(context.Background(), agentsdk.LLMRequest{}) {
				err = e
			}

			var apiErr *anthropicsdk.Error
			require.ErrorAs(t, err, &apiErr)
			assert.Equal(t, tt.status, apiErr.StatusCode)
		})
	}
}
