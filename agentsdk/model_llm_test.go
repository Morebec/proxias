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
		ok    bool
	}{
		{
			name:  "GIVEN the agent's action request THEN it is the assistant's tool call",
			given: agentsdk.Event{ID: "toolu_1", Type: agentsdk.EventTypeActionRequest, Source: agentsdk.NewAgentSource("ada"), Payload: []byte(`{"action_id":"send_email","params":{"to":"joe"}}`)},
			then:  agentsdk.LLMMessage{Role: agentsdk.RoleAssistant, Parts: []agentsdk.LLMPart{{ToolCall: &agentsdk.LLMToolCall{ID: "toolu_1", Name: "send_email", Input: map[string]any{"to": "joe"}}}}},
			ok:    true,
		},
		{
			name:  "GIVEN the agent's action request that is not one THEN it is the assistant's text, tagged",
			given: agentsdk.Event{Type: agentsdk.EventTypeActionRequest, Source: agentsdk.NewAgentSource("ada"), Payload: []byte(`send`)},
			then:  agentsdk.LLMMessage{Role: agentsdk.RoleAssistant, Parts: []agentsdk.LLMPart{{Text: "[action.request from agent:ada] send"}}},
			ok:    true,
		},
		{
			name:  "GIVEN the agent's signed thinking THEN it is given back as it came",
			given: agentsdk.Event{Type: agentsdk.EventTypeThought, Source: agentsdk.NewAgentSource("ada"), Payload: []byte(`{"Thinking":"An email to joe.","Signature":"sig1"}`)},
			then:  agentsdk.LLMMessage{Role: agentsdk.RoleAssistant, Parts: []agentsdk.LLMPart{{Thinking: "An email to joe.", Signature: "sig1"}}},
			ok:    true,
		},
		{
			name:  "GIVEN the agent's thought in words THEN it is the assistant's text",
			given: agentsdk.Event{Type: agentsdk.EventTypeThought, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("An email.")},
			then:  agentsdk.LLMMessage{Role: agentsdk.RoleAssistant, Parts: []agentsdk.LLMPart{{Text: "An email."}}},
			ok:    true,
		},
		{
			name:  "GIVEN the agent's empty message THEN the model is not told it",
			given: agentsdk.Event{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewAgentSource("ada")},
			then:  agentsdk.LLMMessage{Role: agentsdk.RoleAssistant, Parts: []agentsdk.LLMPart{{}}},
		},
		{
			name:  "GIVEN an action result THEN it is the user's tool result",
			given: agentsdk.Event{Type: agentsdk.EventTypeActionResult, Source: agentsdk.NewSystemSource("actions"), CausationID: "toolu_1", Payload: []byte(`{"status":"completed"}`)},
			then:  agentsdk.LLMMessage{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{ToolResult: &agentsdk.LLMToolResult{CallID: "toolu_1", Content: `{"status":"completed"}`}}}},
			ok:    true,
		},
		{
			name:  "GIVEN a person's message THEN it is the user's text",
			given: agentsdk.Event{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("hi")},
			then:  agentsdk.LLMMessage{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "hi"}}},
			ok:    true,
		},
		{
			name:  "GIVEN another agent's message THEN it is the user's text, tagged",
			given: agentsdk.Event{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewAgentSource("rita"), Payload: []byte("hi")},
			then:  agentsdk.LLMMessage{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "[message from agent:rita] hi"}}},
			ok:    true,
		},
		{
			name:  "GIVEN an empty event of another source THEN the model is not told it",
			given: agentsdk.Event{Type: "probe.disk", Source: agentsdk.NewSystemSource("db-1")},
			then:  agentsdk.LLMMessage{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "[probe.disk from system:db-1] "}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, ok := agentsdk.NewLLMMessage(agentsdk.NewAgentSource("ada"), tt.given)

			assert.Equal(t, tt.ok, ok)
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
			name: "GIVEN leading messages of a system source THEN they are the system prompt, and a later one a message",
			given: []agentsdk.Event{
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewSystemSource("app"), Payload: []byte("Be brief.")},
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewSystemSource("app"), Payload: []byte("Be kind.")},
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("hi")},
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewSystemSource("app"), Payload: []byte("Wrap up.")},
			},
			then: agentsdk.LLMRequest{
				System: []string{"Be brief.", "Be kind."},
				Messages: []agentsdk.LLMMessage{
					{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "hi"}}},
					{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "[message from system:app] Wrap up."}}},
				},
			},
		},
		{
			name: "GIVEN the agent's request, its result and its reply THEN they are a tool call, its result and the assistant's text, one message each",
			given: []agentsdk.Event{
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("email joe")},
				{ID: "toolu_1", Type: agentsdk.EventTypeActionRequest, Source: agentsdk.NewAgentSource("ada"), Payload: []byte(`{"action_id":"send_email"}`)},
				{Type: agentsdk.EventTypeActionResult, Source: agentsdk.NewSystemSource("actions"), CausationID: "toolu_1", Payload: []byte(`{"status":"completed"}`)},
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("Sent.")},
			},
			then: agentsdk.LLMRequest{
				Messages: []agentsdk.LLMMessage{
					{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "email joe"}}},
					{Role: agentsdk.RoleAssistant, Parts: []agentsdk.LLMPart{{ToolCall: &agentsdk.LLMToolCall{ID: "toolu_1", Name: "send_email"}}}},
					{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{ToolResult: &agentsdk.LLMToolResult{CallID: "toolu_1", Content: `{"status":"completed"}`}}}},
					{Role: agentsdk.RoleAssistant, Parts: []agentsdk.LLMPart{{Text: "Sent."}}},
				},
			},
		},
		{
			name: "GIVEN events the model is not told THEN they are left out",
			given: []agentsdk.Event{
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("hi")},
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewAgentSource("ada")},
			},
			then: agentsdk.LLMRequest{
				Messages: []agentsdk.LLMMessage{{Role: agentsdk.RoleUser, Parts: []agentsdk.LLMPart{{Text: "hi"}}}},
			},
		},
		{
			name: "GIVEN no events THEN the request is empty",
			then: agentsdk.LLMRequest{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := agentsdk.NewLLMRequestFromSession(agentsdk.NewSession(tt.given...), agentsdk.NewAgentSource("ada"))

			assert.Equal(t, tt.then, req)
		})
	}
}
