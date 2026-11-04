package agentsdk_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias/agentsdk"
)

func TestNewLLMMessage(t *testing.T) {
	tests := []struct {
		name  string
		given agentsdk.Event
		then  agentsdk.LLMMessage
	}{
		{
			name: "GIVEN the agent's thinking, unsigned (an OpenAI-compatible model's) THEN it is given back as it came",
			given: agentsdk.Event{
				Type:    agentsdk.EventTypeThought,
				Source:  agentsdk.NewAgentSource("ada"),
				Payload: []byte(`{"Thinking":"The user greets me."}`),
			},
			then: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{Thinking: "The user greets me."},
				},
			},
		},
		{
			name: "GIVEN the agent's action request THEN it is the assistant's tool call",
			given: agentsdk.Event{
				ID:     "toolu_1",
				Type:   agentsdk.EventTypeActionRequest,
				Source: agentsdk.NewAgentSource("ada"),
				Payload: []byte(
					`{"action_id":"send_email","params":{"to":"joe"}}`,
				),
			},
			then: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{
						ToolCall: &agentsdk.LLMToolCall{
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
			given: agentsdk.Event{
				Type:    agentsdk.EventTypeActionRequest,
				Source:  agentsdk.NewAgentSource("ada"),
				Payload: []byte(`send`),
			},
			then: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{Text: "[action.request from agent:ada] send"},
				},
			},
		},
		{
			name: "GIVEN the agent's signed thinking THEN it is given back as it came",
			given: agentsdk.Event{
				Type:   agentsdk.EventTypeThought,
				Source: agentsdk.NewAgentSource("ada"),
				Payload: []byte(
					`{"Thinking":"An email to joe.","Signature":"sig1"}`,
				),
			},
			then: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{Thinking: "An email to joe.", Signature: "sig1"},
				},
			},
		},
		{
			name: "GIVEN the agent's thought in words THEN it is the assistant's text",
			given: agentsdk.Event{
				Type:    agentsdk.EventTypeThought,
				Source:  agentsdk.NewAgentSource("ada"),
				Payload: []byte("An email."),
			},
			then: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{Text: "An email."},
				},
			},
		},
		{
			name: "GIVEN the agent's empty message THEN it is an empty message",
			given: agentsdk.Event{
				Type:   agentsdk.EventTypeMessage,
				Source: agentsdk.NewAgentSource("ada"),
			},
			then: agentsdk.LLMMessage{
				Role:  agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{{}},
			},
		},
		{
			name: "GIVEN an action result THEN it is the tool's result",
			given: agentsdk.Event{
				Type:        agentsdk.EventTypeActionResult,
				Source:      agentsdk.NewSystemSource("actions"),
				CausationID: "toolu_1",
				Payload:     []byte(`{"status":"completed"}`),
			},
			then: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleTool,
				Parts: []agentsdk.LLMPart{
					{
						ToolResult: &agentsdk.LLMToolResult{
							CallID:  "toolu_1",
							Content: `{"status":"completed"}`,
						},
					},
				},
			},
		},
		{
			name: "GIVEN a person's message THEN it is the user's text",
			given: agentsdk.Event{
				Type:    agentsdk.EventTypeMessage,
				Source:  agentsdk.NewUserSource("john"),
				Payload: []byte("hi"),
			},
			then: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleUser,
				Parts: []agentsdk.LLMPart{
					{Text: "hi"},
				},
			},
		},
		{
			name: "GIVEN another agent's message THEN it is the user's text, tagged",
			given: agentsdk.Event{
				Type:    agentsdk.EventTypeMessage,
				Source:  agentsdk.NewAgentSource("rita"),
				Payload: []byte("hi"),
			},
			then: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleUser,
				Parts: []agentsdk.LLMPart{
					{Text: "[message from agent:rita] hi"},
				},
			},
		},
		{
			name: "GIVEN a system's observation THEN it is the user's, tagged with its type and source: not an instruction",
			given: agentsdk.Event{
				Type:    "probe.disk",
				Source:  agentsdk.NewSystemSource("db-1"),
				Payload: []byte("disk 93% full"),
			},
			then: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleUser,
				Parts: []agentsdk.LLMPart{
					{Text: "[probe.disk from system:db-1] disk 93% full"},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := agentsdk.NewLLMMessageFromEvent(agentsdk.NewAgentSource("ada"), tt.given)

			assert.Equal(t, tt.then, m)
		})
	}
}

func TestNewLLMRequestFromSession(t *testing.T) {
	tests := []struct {
		name  string
		given []agentsdk.Event
		then  agentsdk.LLMRequest
	}{
		{
			name: "GIVEN messages of a system source THEN they are system messages, where they are in the session",
			given: []agentsdk.Event{
				{
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewSystemSource("app"),
					Payload: []byte("Be brief."),
				},
				{
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewSystemSource("app"),
					Payload: []byte("Be kind."),
				},
				{
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("hi"),
				},
				{
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewSystemSource("app"),
					Payload: []byte("Wrap up."),
				},
			},
			then: agentsdk.LLMRequest{
				Messages: []agentsdk.LLMMessage{
					{
						Role: agentsdk.LLMMessageRoleSystem,
						Parts: []agentsdk.LLMPart{
							{Text: "Be brief."},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleSystem,
						Parts: []agentsdk.LLMPart{
							{Text: "Be kind."},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{
							{Text: "hi"},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleSystem,
						Parts: []agentsdk.LLMPart{
							{Text: "Wrap up."},
						},
					},
				},
			},
		},
		{
			name: "GIVEN the agent's request, its result and its reply THEN they are a tool call, its result and the assistant's text, one message each",
			given: []agentsdk.Event{
				{
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("email joe"),
				},
				{
					ID:      "toolu_1",
					Type:    agentsdk.EventTypeActionRequest,
					Source:  agentsdk.NewAgentSource("ada"),
					Payload: []byte(`{"action_id":"send_email"}`),
				},
				{
					Type:        agentsdk.EventTypeActionResult,
					Source:      agentsdk.NewSystemSource("actions"),
					CausationID: "toolu_1",
					Payload:     []byte(`{"status":"completed"}`),
				},
				{
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewAgentSource("ada"),
					Payload: []byte("Sent."),
				},
			},
			then: agentsdk.LLMRequest{
				Messages: []agentsdk.LLMMessage{
					{
						Role: agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{
							{Text: "email joe"},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{
								ToolCall: &agentsdk.LLMToolCall{
									ID:   "toolu_1",
									Name: "send_email",
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
									Content: `{"status":"completed"}`,
								},
							},
						},
					},
					{
						Role: agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{
							{Text: "Sent."},
						},
					},
				},
			},
		},
		{
			name: "GIVEN an empty event THEN it is an empty message, which the adapters leave out",
			given: []agentsdk.Event{
				{
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("hi"),
				},
				{
					Type:   agentsdk.EventTypeMessage,
					Source: agentsdk.NewAgentSource("ada"),
				},
			},
			then: agentsdk.LLMRequest{
				Messages: []agentsdk.LLMMessage{
					{
						Role: agentsdk.LLMMessageRoleUser,
						Parts: []agentsdk.LLMPart{
							{Text: "hi"},
						},
					},
					{
						Role:  agentsdk.LLMMessageRoleAssistant,
						Parts: []agentsdk.LLMPart{{}},
					},
				},
			},
		},
		{
			name: "GIVEN no events THEN the request is empty",
			then: agentsdk.LLMRequest{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := agentsdk.NewLLMRequestFromSession(agentsdk.NewAgentSource("ada"), agentsdk.NewSession(tt.given...))

			assert.Equal(t, tt.then, req)
		})
	}
}

func TestNewEventsFromLLMMessage(t *testing.T) {
	tests := []struct {
		name  string
		given agentsdk.LLMMessage
		// then is each event's type, source, and payload; its ID too when it is
		// a request's (the call's) or the message's.
		then []string
	}{
		{
			name: "GIVEN text THEN it is a message of the agent",
			given: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{Text: "Hello John."},
				},
			},
			then: []string{
				"message agent:ada Hello John.",
			},
		},
		{
			name: "GIVEN no text THEN it is an empty message: silence",
			given: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{Text: ""},
				},
			},
			then: []string{
				"message agent:ada ",
			},
		},
		{
			name: "GIVEN thinking, signed and redacted, then text THEN the thoughts, kept whole, come before the message",
			given: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
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
			given: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{
						ToolCall: &agentsdk.LLMToolCall{
							ID:   "toolu_1",
							Name: "send_email",
							Input: map[string]any{
								"to": "joe",
							},
						},
					},
					{
						ToolCall: &agentsdk.LLMToolCall{
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
			given: agentsdk.LLMMessage{
				ID:   "m1",
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{Text: "Hello John."},
				},
			},
			then: []string{
				"message agent:ada m1 Hello John.",
			},
		},
		{
			name: "GIVEN a message of an ID, with a tool call THEN its text is a thought of that ID",
			given: agentsdk.LLMMessage{
				ID:   "m1",
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{
						ToolCall: &agentsdk.LLMToolCall{
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
			given: agentsdk.LLMMessage{
				Role: agentsdk.LLMMessageRoleAssistant,
				Parts: []agentsdk.LLMPart{
					{
						Thinking:  "John wants an email.",
						Signature: "sig",
					},
					{
						ToolCall: &agentsdk.LLMToolCall{
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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			events := agentsdk.NewEventsFromLLMMessage(agentsdk.NewAgentSource("ada"), tt.given)

			got := make([]string, 0, len(events))
			for _, e := range events {
				assert.NotEmpty(t, e.ID, "the events are whole")
				assert.False(t, e.Date.IsZero(), "the events are dated")
				line := string(e.Type) + " " + e.Source.String() + " "
				if e.Type == agentsdk.EventTypeActionRequest || e.ID == tt.given.ID {
					line += string(e.ID) + " "
				}
				got = append(got, line+e.Text())
			}
			assert.Equal(t, tt.then, got)
		})
	}
}
