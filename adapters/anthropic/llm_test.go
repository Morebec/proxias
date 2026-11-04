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

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
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
		when     proxias.LLMRequest
		then     []proxias.LLMChunk
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
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{Role: proxias.LLMMessageRoleUser, Parts: []proxias.LLMPart{{Text: "hello"}}},
				{Role: proxias.LLMMessageRoleAssistant, Parts: []proxias.LLMPart{{Thinking: "The user greets me."}, {Text: "Hi."}}},
				{Role: proxias.LLMMessageRoleUser, Parts: []proxias.LLMPart{{Text: "again"}}},
			}},
			then: []proxias.LLMChunk{{Text: "ok"}, {StopReason: proxias.StopReasonEndTurn}},
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
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{
					Role:  proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{{Text: "what time is it?"}},
				},
			}},
			thinking: anthropicsdk.ThinkingConfigParamUnion{
				OfAdaptive: &anthropicsdk.ThinkingConfigAdaptiveParam{
					Display: anthropicsdk.ThinkingConfigAdaptiveDisplaySummarized,
				},
			},
			requested: `{"type":"adaptive","display":"summarized"}`,
			then: []proxias.LLMChunk{
				{Thinking: "The user wants the time.", Signature: "sig1"},
				{RedactedThinking: "secret"},
				{
					ToolCall: &proxias.LLMToolCall{
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
			when: proxias.LLMRequest{
				Tools: []proxias.ActionDescriptor{{ID: "get_time"}},
				Messages: []proxias.LLMMessage{
					{
						Role:  proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{{Text: "what time is it?"}},
					},
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{
								Thinking:  "The user wants the time.",
								Signature: "sig1",
							},
							{RedactedThinking: "secret"},
							{
								ToolCall: &proxias.LLMToolCall{
									ID:   "toolu_1",
									Name: "get_time",
								},
							},
						},
					},
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{
								ToolResult: &proxias.LLMToolResult{
									CallID:  "toolu_1",
									Content: "noon",
								},
							},
						},
					},
				},
			},
			then: []proxias.LLMChunk{
				{Text: "It is noon."},
				{StopReason: proxias.StopReasonEndTurn},
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
			when: proxias.LLMRequest{

				Tools: []proxias.ActionDescriptor{
					{
						ID:          "send_email",
						Description: "Sends an email.",
						InputSchema: map[string]any{
							"type": "object", "properties": map[string]any{"to": map[string]any{"type": "string"}}, "required": []string{"to"},
						},
					},
				},
				Messages: []proxias.LLMMessage{
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
						Parts: []proxias.LLMPart{{Text: "email joe"}},
					},
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{
								ToolCall: &proxias.LLMToolCall{
									ID:    "toolu_0",
									Name:  "send_email",
									Input: map[string]any{"to": "joe"},
								},
							},
						},
					},
					{
						Role: proxias.LLMMessageRoleTool,
						Parts: []proxias.LLMPart{
							{
								ToolResult: &proxias.LLMToolResult{
									CallID:  "toolu_0",
									Content: "sent",
								},
							},
						},
					},
				},
			},
			then: []proxias.LLMChunk{
				{Text: "An email."},
				{
					ToolCall: &proxias.LLMToolCall{
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
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{
					Role:  proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{{Text: "email joe"}},
				},
				{
					Role: proxias.LLMMessageRoleAssistant,
					Parts: []proxias.LLMPart{
						{Text: "An email."},
						{
							ToolCall: &proxias.LLMToolCall{
								ID:   "toolu_0",
								Name: "send_email",
							},
						},
					},
				},
				{
					Role: proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{
						{
							ToolResult: &proxias.LLMToolResult{
								CallID:  "toolu_0",
								Content: "sent",
							},
						},
					},
				},
				{
					Role:  proxias.LLMMessageRoleUser,
					Parts: []proxias.LLMPart{{Text: "thanks"}},
				},
			}},
			then: []proxias.LLMChunk{
				{Text: "Sent."},
				{StopReason: proxias.StopReasonEndTurn},
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
			when: proxias.LLMRequest{
				Messages: []proxias.LLMMessage{
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "hello"},
						},
					},
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
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
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{Text: "Hi."},
						},
					},
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "thanks"},
						},
					},
				},
			},
			then: []proxias.LLMChunk{
				{Text: "Hi."},
				{StopReason: proxias.StopReasonEndTurn},
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
			when: proxias.LLMRequest{
				Tools: []proxias.ActionDescriptor{
					{ID: "get_time"},
				},
				Messages: []proxias.LLMMessage{
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "hello"},
						},
					},
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{
								Thinking:  "The user greets me.",
								Signature: "sig0",
							},
							{RedactedThinking: "old"},
							{Text: "Hi."},
						},
					},
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "what time is it?"},
						},
					},
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{
								Thinking:  "The user wants the time.",
								Signature: "sig1",
							},
						},
					},
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{
								ToolCall: &proxias.LLMToolCall{
									ID:   "toolu_1",
									Name: "get_time",
								},
							},
						},
					},
					{
						Role: proxias.LLMMessageRoleTool,
						Parts: []proxias.LLMPart{
							{
								ToolResult: &proxias.LLMToolResult{
									CallID:  "toolu_1",
									Content: "noon",
								},
							},
						},
					},
				},
			},
			then: []proxias.LLMChunk{
				{Text: "It is noon."},
				{StopReason: proxias.StopReasonEndTurn},
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
			when: proxias.LLMRequest{
				Tools: []proxias.ActionDescriptor{
					{ID: "get_time"},
				},
				Messages: []proxias.LLMMessage{
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{
								Thinking:  "The user greets me.",
								Signature: "sig0",
							},
							{Text: "It is noon."},
						},
					},
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "thanks"},
						},
					},
				},
			},
			then: []proxias.LLMChunk{
				{Text: "You're welcome."},
				{StopReason: proxias.StopReasonEndTurn},
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
					Role:  proxias.LLMMessageRoleAssistant,
					Parts: []proxias.LLMPart{{Text: "Hello."}},
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
			messages: `[
				{"role":"user","content":[{"type":"text","text":"hi"},{"type":"text","text":"still there?"}]}
			]`,
		},
		{
			name:   "GIVEN a conversation that ends on the agent THEN a user turn ends it",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"ok"}]}`,
			when: proxias.LLMRequest{Messages: []proxias.LLMMessage{
				{
					Role:  proxias.LLMMessageRoleAssistant,
					Parts: []proxias.LLMPart{{Text: "I waited."}},
				},
			}},
			then: []proxias.LLMChunk{
				{Text: "ok"},
				{StopReason: proxias.StopReasonEndTurn},
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
			then: []proxias.LLMChunk{
				{Text: "ok"},
				{StopReason: proxias.StopReasonEndTurn},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"(nothing happened yet)"}]}
			]`,
		}, {
			name:   "GIVEN Claude stops at the limit of tokens THEN the answer so far, and the length",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"max_tokens","usage":{},"content":[{"type":"text","text":"Once upon"}]}`,
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
			messages: `[{"role":"user","content":[{"type":"text","text":"tell me a story"}]}]`,
		},
		{
			name:   "GIVEN Claude refuses THEN the content filter",
			status: http.StatusOK,
			answer: `{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"refusal","usage":{},"content":[]}`,
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
			messages: `[{"role":"user","content":[{"type":"text","text":"something harmful"}]}]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var request map[string]any
			client := mockClaude(t, tt.status, tt.answer, &request)

			var chunks []proxias.LLMChunk
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
			for _, e := range anthropic.NewLLM(client, "m").Complete(context.Background(), proxias.LLMRequest{}) {
				err = e
			}

			var apiErr *anthropicsdk.Error
			require.ErrorAs(t, err, &apiErr)
			assert.Equal(t, tt.status, apiErr.StatusCode)
		})
	}
}
