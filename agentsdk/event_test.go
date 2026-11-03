package agentsdk_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
)

func TestEventSourceString(t *testing.T) {
	tests := []struct {
		name  string
		given agentsdk.EventSource
		then  string
	}{
		{
			name:  "GIVEN a source with no metadata THEN it is its type and ID in string format",
			given: agentsdk.NewUserSource("john"),
			then:  "user:john",
		},
		{
			name: "GIVEN a source with metadata THEN the metadata follows",
			given: agentsdk.EventSource{
				Type:     agentsdk.EventSourceTypeSystem,
				ID:       "db-1",
				Metadata: "disk",
			},
			then: "system:db-1:disk",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.then, tt.given.String())
		})
	}
}

func TestEventHasSourceType(t *testing.T) {
	tests := []struct {
		name  string
		given agentsdk.Event
		when  agentsdk.EventSourceType
		then  bool
	}{
		{
			name:  "GIVEN an event of a person WHEN asked for a person THEN it has the source type",
			given: agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hi"),
			when:  agentsdk.EventSourceTypeUser,
			then:  true,
		},
		{
			name:  "GIVEN an event of a system WHEN asked for an agent THEN it has not",
			given: agentsdk.NewMessageEvent(agentsdk.NewSystemSource("clock"), "It is noon."),
			when:  agentsdk.EventSourceTypeAgent,
			then:  false,
		},
		{
			name:  "GIVEN an event of no source WHEN asked for a system THEN it has not",
			given: agentsdk.Event{},
			when:  agentsdk.EventSourceTypeSystem,
			then:  false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.then, tt.given.HasSourceType(tt.when))
		})
	}
}

func TestNewMessageEvent(t *testing.T) {
	e := agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hello")

	assert.NotEmpty(t, e.ID)
	assert.False(t, e.Date.IsZero())
	assert.Equal(t, agentsdk.EventTypeMessage, e.Type)
	assert.Equal(t, agentsdk.NewUserSource("john"), e.Source)
	assert.Equal(t, "hello", e.Text())
}

func TestNewActionEvents(t *testing.T) {
	tests := []struct {
		name string
		when func() (agentsdk.Event, error)
		// then is the event's type, source and payload; err, when set, its
		// error.
		then []string
		err  string
	}{
		{
			name: "GIVEN a request THEN it is the request event of the source",
			when: func() (agentsdk.Event, error) {
				return agentsdk.NewActionExecutionRequestEvent(
					agentsdk.NewAgentSource("ada"),
					agentsdk.ActionExecutionRequest{
						ActionID: "send_email",
						Params: map[string]any{
							"to": "joe",
						},
					},
				)
			},
			then: []string{
				"action.request",
				"agent:ada",
				`{"action_id":"send_email","params":{"to":"joe"}}`,
			},
		},
		{
			name: "GIVEN a request whose params are not JSON THEN it is an error",
			when: func() (agentsdk.Event, error) {
				return agentsdk.NewActionExecutionRequestEvent(
					agentsdk.NewAgentSource("ada"),
					agentsdk.ActionExecutionRequest{
						ActionID: "send_email",
						Params: map[string]any{
							"to": make(chan int),
						},
					},
				)
			},
			err: "send_email",
		},
		{
			name: "GIVEN a result THEN it is the result event of the source",
			when: func() (agentsdk.Event, error) {
				return agentsdk.NewActionExecutionResultEvent(
					agentsdk.NewSystemSource("actions"),
					agentsdk.ActionExecutionResult{
						ActionID: "send_email",
						Status:   "completed",
					},
				)
			},
			then: []string{
				"action.result",
				"system:actions",
				`{"action_id":"send_email","status":"completed"}`,
			},
		},
		{
			name: "GIVEN a result whose data is not JSON THEN it is an error",
			when: func() (agentsdk.Event, error) {
				return agentsdk.NewActionExecutionResultEvent(
					agentsdk.NewSystemSource("actions"),
					agentsdk.ActionExecutionResult{
						ActionID: "send_email",
						Data: map[string]any{
							"sent": make(chan int),
						},
					},
				)
			},
			err: "send_email",
		},
		{
			name: "GIVEN a chunk of thinking THEN it is the thought event of the source, its thinking whole",
			when: func() (agentsdk.Event, error) {
				chunk := agentsdk.LLMChunk{
					Text:      "ignored",
					Thinking:  "The user greets me.",
					Signature: "sig",
				}
				return agentsdk.NewThoughtEvent(agentsdk.NewAgentSource("ada"), chunk), nil
			},
			then: []string{
				"cognition.thought",
				"agent:ada",
				`{"Text":"","ToolCall":null,"Thinking":"The user greets me.","Signature":"sig","RedactedThinking":""}`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := tt.when()

			if tt.err != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.err)
				return
			}
			require.NoError(t, err)
			assert.NotEmpty(t, e.ID)
			assert.False(t, e.Date.IsZero())
			assert.Equal(t, tt.then, []string{string(e.Type), e.Source.String(), e.Text()})
		})
	}
}
