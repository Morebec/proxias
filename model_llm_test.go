package proxias_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias"
)

func TestNewLLMMessage(t *testing.T) {
	tests := []struct {
		name  string
		given proxias.Event
		then  proxias.LLMMessage
	}{
		{
			name: "GIVEN the agent's thinking, unsigned (an OpenAI-compatible model's) THEN it is given back as it came",
			given: proxias.Event{
				Type:    proxias.EventTypeThought,
				Source:  proxias.NewAgentSource("ada"),
				Payload: []byte(`{"Thinking":"The user greets me."}`),
			},
			then: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{Thinking: "The user greets me."},
				},
			},
		},
		{
			name: "GIVEN the agent's action request THEN it is the assistant's tool call",
			given: proxias.Event{
				ID:     "toolu_1",
				Type:   proxias.EventTypeActionRequest,
				Source: proxias.NewAgentSource("ada"),
				Payload: []byte(
					`{"action_id":"send_email","params":{"to":"joe"}}`,
				),
			},
			then: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{
						ToolCall: &proxias.LLMToolCall{
							ID:    "toolu_1",
							Name:  "send_email",
							Input: map[string]any{"to": "joe"},
						},
					},
				},
			},
		},
		{
			name: "GIVEN the agent's action request that is not one THEN it is the assistant's text, tagged",
			given: proxias.Event{
				Type:    proxias.EventTypeActionRequest,
				Source:  proxias.NewAgentSource("ada"),
				Payload: []byte(`send`),
			},
			then: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{Text: "[action.request from agent:ada] send"},
				},
			},
		},
		{
			name: "GIVEN the agent's signed thinking THEN it is given back as it came",
			given: proxias.Event{
				Type:   proxias.EventTypeThought,
				Source: proxias.NewAgentSource("ada"),
				Payload: []byte(
					`{"Thinking":"An email to joe.","Signature":"sig1"}`,
				),
			},
			then: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{Thinking: "An email to joe.", Signature: "sig1"},
				},
			},
		},
		{
			name: "GIVEN the agent's thought in words THEN it is the assistant's text",
			given: proxias.Event{
				Type:    proxias.EventTypeThought,
				Source:  proxias.NewAgentSource("ada"),
				Payload: []byte("An email."),
			},
			then: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{Text: "An email."},
				},
			},
		},
		{
			name: "GIVEN the agent's thought of an opaque block THEN the block, given back as it came",
			given: proxias.Event{
				Type:    proxias.EventTypeThought,
				Source:  proxias.NewAgentSource("ada"),
				Payload: []byte(`{"Opaque":{"type":"server_tool_use"}}`),
			},
			then: proxias.LLMMessage{
				Role:  proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{{Opaque: json.RawMessage(`{"type":"server_tool_use"}`)}},
			},
		},
		{
			name: "GIVEN the agent's empty message THEN it is an empty message",
			given: proxias.Event{
				Type:   proxias.EventTypeMessage,
				Source: proxias.NewAgentSource("ada"),
			},
			then: proxias.LLMMessage{
				Role:  proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{{}},
			},
		},
		{
			name: "GIVEN an action result THEN it is the tool's result",
			given: proxias.Event{
				Type:        proxias.EventTypeActionResult,
				Source:      proxias.NewSystemSource("actions"),
				CausationID: "toolu_1",
				Payload:     []byte(`{"status":"completed"}`),
			},
			then: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleTool,
				Parts: []proxias.LLMPart{
					{
						ToolResult: &proxias.LLMToolResult{
							CallID:  "toolu_1",
							Content: `{"status":"completed"}`,
						},
					},
				},
			},
		},
		{
			name: "GIVEN a person's message THEN it is the user's text",
			given: proxias.Event{
				Type:    proxias.EventTypeMessage,
				Source:  proxias.NewUserSource("john"),
				Payload: []byte("hi"),
			},
			then: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleUser,
				Parts: []proxias.LLMPart{
					{Text: "hi"},
				},
			},
		},
		{
			name: "GIVEN another agent's message THEN it is the user's text, tagged",
			given: proxias.Event{
				Type:    proxias.EventTypeMessage,
				Source:  proxias.NewAgentSource("rita"),
				Payload: []byte("hi"),
			},
			then: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleUser,
				Parts: []proxias.LLMPart{
					{Text: "[message from agent:rita] hi"},
				},
			},
		},
		{
			name: "GIVEN a system's observation THEN it is the user's, tagged with its type and source: not an instruction",
			given: proxias.Event{
				Type:    "probe.disk",
				Source:  proxias.NewSystemSource("db-1"),
				Payload: []byte("disk 93% full"),
			},
			then: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleUser,
				Parts: []proxias.LLMPart{
					{Text: "[probe.disk from system:db-1] disk 93% full"},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := proxias.NewLLMMessageFromEvent(proxias.NewAgentSource("ada"), tt.given)

			assert.Equal(t, tt.then, m)
		})
	}
}

func TestNewLLMRequestFromSession(t *testing.T) {
	tests := []struct {
		name  string
		given []proxias.Event
		then  proxias.LLMRequest
	}{
		{
			name: "GIVEN messages of a system source THEN they are system messages, where they are in the session",
			given: []proxias.Event{
				{
					Type:    proxias.EventTypeMessage,
					Source:  proxias.NewSystemSource("app"),
					Payload: []byte("Be brief."),
				},
				{
					Type:    proxias.EventTypeMessage,
					Source:  proxias.NewSystemSource("app"),
					Payload: []byte("Be kind."),
				},
				{
					Type:    proxias.EventTypeMessage,
					Source:  proxias.NewUserSource("john"),
					Payload: []byte("hi"),
				},
				{
					Type:    proxias.EventTypeMessage,
					Source:  proxias.NewSystemSource("app"),
					Payload: []byte("Wrap up."),
				},
			},
			then: proxias.LLMRequest{
				Messages: []proxias.LLMMessage{
					{
						Role: proxias.LLMMessageRoleSystem,
						Parts: []proxias.LLMPart{
							{Text: "Be brief."},
						},
					},
					{
						Role: proxias.LLMMessageRoleSystem,
						Parts: []proxias.LLMPart{
							{Text: "Be kind."},
						},
					},
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "hi"},
						},
					},
					{
						Role: proxias.LLMMessageRoleSystem,
						Parts: []proxias.LLMPart{
							{Text: "Wrap up."},
						},
					},
				},
			},
		},
		{
			name: "GIVEN the agent's request, its result and its reply THEN they are a tool call, its result and the assistant's text, one message each",
			given: []proxias.Event{
				{
					Type:    proxias.EventTypeMessage,
					Source:  proxias.NewUserSource("john"),
					Payload: []byte("email joe"),
				},
				{
					ID:      "toolu_1",
					Type:    proxias.EventTypeActionRequest,
					Source:  proxias.NewAgentSource("ada"),
					Payload: []byte(`{"action_id":"send_email"}`),
				},
				{
					Type:        proxias.EventTypeActionResult,
					Source:      proxias.NewSystemSource("actions"),
					CausationID: "toolu_1",
					Payload:     []byte(`{"status":"completed"}`),
				},
				{
					Type:    proxias.EventTypeMessage,
					Source:  proxias.NewAgentSource("ada"),
					Payload: []byte("Sent."),
				},
			},
			then: proxias.LLMRequest{
				Messages: []proxias.LLMMessage{
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "email joe"},
						},
					},
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{
								ToolCall: &proxias.LLMToolCall{
									ID:   "toolu_1",
									Name: "send_email",
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
									Content: `{"status":"completed"}`,
								},
							},
						},
					},
					{
						Role: proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{
							{Text: "Sent."},
						},
					},
				},
			},
		},
		{
			name: "GIVEN an empty event THEN it is an empty message, which the adapters leave out",
			given: []proxias.Event{
				{
					Type:    proxias.EventTypeMessage,
					Source:  proxias.NewUserSource("john"),
					Payload: []byte("hi"),
				},
				{
					Type:   proxias.EventTypeMessage,
					Source: proxias.NewAgentSource("ada"),
				},
			},
			then: proxias.LLMRequest{
				Messages: []proxias.LLMMessage{
					{
						Role: proxias.LLMMessageRoleUser,
						Parts: []proxias.LLMPart{
							{Text: "hi"},
						},
					},
					{
						Role:  proxias.LLMMessageRoleAssistant,
						Parts: []proxias.LLMPart{{}},
					},
				},
			},
		},
		{
			name: "GIVEN no events THEN the request is empty",
			then: proxias.LLMRequest{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := proxias.NewLLMRequestFromSession(proxias.NewAgentSource("ada"), proxias.NewSession(tt.given...))

			assert.Equal(t, tt.then, req)
		})
	}
}

func TestNewEventsFromLLMMessage(t *testing.T) {
	tests := []struct {
		name  string
		given proxias.LLMMessage
		// then is each event's type, source, and payload; its ID too when it is
		// a request's (the call's) or the message's.
		then []string
	}{
		{
			name: "GIVEN text THEN it is a message of the agent",
			given: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{Text: "Hello John."},
				},
			},
			then: []string{
				"message agent:ada Hello John.",
			},
		},
		{
			name: "GIVEN no text THEN it is an empty message: silence",
			given: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{Text: ""},
				},
			},
			then: []string{
				"message agent:ada ",
			},
		},
		{
			name: "GIVEN thinking, signed and redacted, then text THEN the thoughts, kept whole, come before the message",
			given: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{
						Thinking:  "The user greets me.",
						Signature: "sig",
					},
					{RedactedThinking: "secret"},
					{Text: "Hello John."},
				},
			},
			then: []string{
				`cognition.thought agent:ada {"Text":"","ToolCall":null,"Thinking":"The user greets me.","Signature":"sig","RedactedThinking":""}`,
				`cognition.thought agent:ada {"Text":"","ToolCall":null,"Thinking":"","Signature":"","RedactedThinking":"secret"}`,
				"message agent:ada Hello John.",
			},
		},
		{
			name: "GIVEN text and tool calls THEN the text is a thought, before the requests of the calls' IDs",
			given: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{
						ToolCall: &proxias.LLMToolCall{
							ID:   "toolu_1",
							Name: "send_email",
							Input: map[string]any{
								"to": "joe",
							},
						},
					},
					{
						ToolCall: &proxias.LLMToolCall{
							ID:   "toolu_2",
							Name: "page_oncall",
						},
					},
					{Text: "An email, and a page."},
				},
			},
			then: []string{
				"cognition.thought agent:ada An email, and a page.",
				`action.request agent:ada toolu_1 {"action_id":"send_email","params":{"to":"joe"}}`,
				`action.request agent:ada toolu_2 {"action_id":"page_oncall"}`,
			},
		},
		{
			name: "GIVEN a message of an ID THEN its text is a message of that ID: the event its chunks were streamed for",
			given: proxias.LLMMessage{
				ID:   "m1",
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{Text: "Hello John."},
				},
			},
			then: []string{
				"message agent:ada m1 Hello John.",
			},
		},
		{
			name: "GIVEN a message of an ID, with a tool call THEN its text is a thought of that ID",
			given: proxias.LLMMessage{
				ID:   "m1",
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{
						ToolCall: &proxias.LLMToolCall{
							ID:   "toolu_1",
							Name: "send_email",
						},
					},
					{Text: "An email."},
				},
			},
			then: []string{
				"cognition.thought agent:ada m1 An email.",
				`action.request agent:ada toolu_1 {"action_id":"send_email"}`,
			},
		},
		{
			name: "GIVEN thinking and a tool call, without text THEN the thought, then the request",
			given: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{
						Thinking:  "John wants an email.",
						Signature: "sig",
					},
					{
						ToolCall: &proxias.LLMToolCall{
							ID:   "toolu_1",
							Name: "send_email",
						},
					},
					{Text: ""},
				},
			},
			then: []string{
				`cognition.thought agent:ada {"Text":"","ToolCall":null,"Thinking":"John wants an email.","Signature":"sig","RedactedThinking":""}`,
				`action.request agent:ada toolu_1 {"action_id":"send_email"}`,
			},
		},
		{
			name: "GIVEN an opaque block and a tool call THEN the block is a thought, kept whole, then the request",
			given: proxias.LLMMessage{
				Role: proxias.LLMMessageRoleAssistant,
				Parts: []proxias.LLMPart{
					{Opaque: json.RawMessage(`{"type":"server_tool_use"}`)},
					{
						ToolCall: &proxias.LLMToolCall{
							ID:   "toolu_1",
							Name: "send_email",
						},
					},
				},
			},
			then: []string{
				`cognition.thought agent:ada {"Text":"","ToolCall":null,"Thinking":"","Signature":"","RedactedThinking":"","Opaque":{"type":"server_tool_use"}}`,
				`action.request agent:ada toolu_1 {"action_id":"send_email"}`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			events := proxias.NewEventsFromLLMMessage(proxias.NewAgentSource("ada"), tt.given)

			got := make([]string, 0, len(events))
			for _, e := range events {
				assert.NotEmpty(t, e.ID, "the events are whole")
				assert.False(t, e.Date.IsZero(), "the events are dated")
				line := string(e.Type) + " " + e.Source.String() + " "
				if e.Type == proxias.EventTypeActionRequest || e.ID == tt.given.ID {
					line += string(e.ID) + " "
				}
				got = append(got, line+e.Text())
			}
			assert.Equal(t, tt.then, got)
		})
	}
}
