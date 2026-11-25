package proxias_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func TestEventSourceString(t *testing.T) {
	tests := []struct {
		name  string
		given proxias.EventSource
		then  string
	}{
		{
			name:  "GIVEN a source with no metadata THEN it is its type and ID in string format",
			given: proxias.NewUserSource("john"),
			then:  "user:john",
		},
		{
			name: "GIVEN a source with metadata THEN the metadata follows",
			given: proxias.EventSource{
				Type:     proxias.EventSourceTypeSystem,
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
		given proxias.Event
		when  proxias.EventSourceType
		then  bool
	}{
		{
			name:  "GIVEN an event of a person WHEN asked for a person THEN it has the source type",
			given: proxias.NewMessageEvent(proxias.NewUserSource("john"), "hi"),
			when:  proxias.EventSourceTypeUser,
			then:  true,
		},
		{
			name:  "GIVEN an event of a system WHEN asked for an agent THEN it has not",
			given: proxias.NewMessageEvent(proxias.NewSystemSource("clock"), "It is noon."),
			when:  proxias.EventSourceTypeAgent,
			then:  false,
		},
		{
			name:  "GIVEN an event of no source WHEN asked for a system THEN it has not",
			given: proxias.Event{},
			when:  proxias.EventSourceTypeSystem,
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
	e := proxias.NewMessageEvent(proxias.NewUserSource("john"), "hello")

	assert.NotEmpty(t, e.ID)
	assert.False(t, e.Date.IsZero())
	assert.Equal(t, proxias.EventTypeMessage, e.Type)
	assert.Equal(t, proxias.NewUserSource("john"), e.Source)
	assert.Equal(t, "hello", e.Text())
	assert.Empty(t, e.Files)
}

func TestNewMessageEventWithFiles(t *testing.T) {
	// Given
	photo := proxias.File{Name: "photo.png", MediaType: "image/png", Content: []byte("png")}

	// When
	e := proxias.NewMessageEvent(proxias.NewUserSource("john"), "look", photo)

	// Then
	assert.Equal(t, "look", e.Text())
	assert.Equal(t, []proxias.File{photo}, e.Files)
}

func TestNewActionExecutionResultEventWithFiles(t *testing.T) {
	// Given
	photo := proxias.File{Name: "photo.png", MediaType: "image/png", Content: []byte("png")}

	// When
	e, err := proxias.NewActionExecutionResultEvent(proxias.NewSystemSource("actions"), proxias.ActionExecutionResult{
		ActionID: "read_file",
		Status:   "completed",
		Files:    []proxias.File{photo},
	})

	// Then
	require.NoError(t, err)
	assert.JSONEq(t, `{"action_id":"read_file","status":"completed"}`, e.Text())
	assert.Equal(t, []proxias.File{photo}, e.Files)
}

func TestNewActionEvents(t *testing.T) {
	tests := []struct {
		name string
		when func() (proxias.Event, error)
		// then is the event's type, source and payload; err, when set, its
		// error.
		then []string
		err  string
	}{
		{
			name: "GIVEN a request THEN it is the request event of the source",
			when: func() (proxias.Event, error) {
				return proxias.NewActionExecutionRequestEvent(
					proxias.NewAgentSource("ada"),
					proxias.ActionExecutionRequest{
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
			when: func() (proxias.Event, error) {
				return proxias.NewActionExecutionRequestEvent(
					proxias.NewAgentSource("ada"),
					proxias.ActionExecutionRequest{
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
			when: func() (proxias.Event, error) {
				return proxias.NewActionExecutionResultEvent(
					proxias.NewSystemSource("actions"),
					proxias.ActionExecutionResult{
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
			when: func() (proxias.Event, error) {
				return proxias.NewActionExecutionResultEvent(
					proxias.NewSystemSource("actions"),
					proxias.ActionExecutionResult{
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
			when: func() (proxias.Event, error) {
				chunk := proxias.LLMChunk{
					Text:      "ignored",
					Thinking:  "The user greets me.",
					Signature: "sig",
				}
				return proxias.NewThoughtEvent(proxias.NewAgentSource("ada"), chunk), nil
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
