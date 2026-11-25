package proxias

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// EventID identifies an event.
type EventID string

// EventType is what an event is to the agent: a message, a thought, an action
// it requests, the result of one...
type EventType string

const (
	// EventTypeMessage represents a message sent by a given source.
	EventTypeMessage EventType = "message"

	// EventTypeThought represents a thought emitted by an event source (mostly
	// agents for chain of thought).
	EventTypeThought EventType = "cognition.thought"

	// EventTypeActionRequest is an action the agent asks its execution layer
	// for. Its payload is an
	// ActionExecutionRequest, as JSON.
	EventTypeActionRequest EventType = "action.request"

	// EventTypeActionResult is the result of an action request, its CausationID
	// the ID of the request. Its payload is
	// an ActionExecutionResult, as JSON.
	EventTypeActionResult EventType = "action.result"
)

// EventSourceType is the kind of source an event comes from.
type EventSourceType string

const (
	// EventSourceTypeUser is a person.
	EventSourceTypeUser EventSourceType = "user"
	// EventSourceTypeAgent is an agent: the one cogitating, or another.
	EventSourceTypeAgent EventSourceType = "agent"
	// EventSourceTypeSystem is anything else: an action runner, a clock, a
	// probe...
	EventSourceTypeSystem EventSourceType = "system"
)

// EventSource indicates from where an event originated.
// This can be a user, an agent, a tool call, a system's state, etc.
type EventSource struct {
	Type     EventSourceType
	ID       string
	Metadata string
}

// NewUserSource is the person id.
func NewUserSource(id string) EventSource {
	return EventSource{Type: EventSourceTypeUser, ID: id}
}

// NewAgentSource is the agent id.
func NewAgentSource(id string) EventSource {
	return EventSource{Type: EventSourceTypeAgent, ID: id}
}

// NewSystemSource is the system id.
func NewSystemSource(id string) EventSource {
	return EventSource{Type: EventSourceTypeSystem, ID: id}
}

// String is the source as "type:id", then ":metadata" when there is some.
func (s EventSource) String() string {
	str := string(s.Type) + ":" + s.ID
	if s.Metadata != "" {
		str = str + ":" + s.Metadata
	}
	return str
}

// Event is an entry of the agent's experience: something it perceived, thought,
// intended or said. Events are chained with ParentID, the event they follow,
// forming a reasoning trace.
type Event struct {
	ID EventID
	// ParentID is the event this one follows in the session, "" for the first.
	ParentID EventID
	// CausationID is the event that caused this one: the action request of an
	// action result.
	CausationID EventID
	Type        EventType
	Source      EventSource
	Payload     []byte
	// Files is the files a message holds, nil for none.
	Files []File
	Date  time.Time
}

// File is a file a message holds: a picture, a document... What a model reads
// of it is its adapter's choice, by its media type.
type File struct {
	Name      string
	MediaType string
	Content   []byte
}

// NewEvent returns an event of a new ID, dated now.
func NewEvent(t EventType, source EventSource, payload []byte) Event {
	return Event{
		ID:      EventID(uuid.NewString()),
		Type:    t,
		Source:  source,
		Payload: payload,
		Date:    time.Now(),
	}
}

// NewMessageEvent returns a message of the source, holding the files.
func NewMessageEvent(source EventSource, text string, files ...File) Event {
	e := NewEvent(EventTypeMessage, source, []byte(text))
	if len(files) > 0 {
		e.Files = files
	}
	return e
}

// NewActionExecutionRequestEvent returns the event of the request, of the
// source: an error when its params are not JSON.
func NewActionExecutionRequestEvent(source EventSource, req ActionExecutionRequest) (Event, error) {
	payload, err := json.Marshal(req)
	if err != nil {
		return Event{}, fmt.Errorf("cannot marshall params of %s: %w", req.ActionID, err)
	}

	return NewEvent(EventTypeActionRequest, source, payload), nil
}

// NewActionExecutionResultEvent returns the event of the result, of the
// source: an error when its data is not JSON.
func NewActionExecutionResultEvent(source EventSource, res ActionExecutionResult) (Event, error) {
	payload, err := json.Marshal(res)
	if err != nil {
		return Event{}, fmt.Errorf("the data of %s is not JSON: %w", res.ActionID, err)
	}

	return NewEvent(EventTypeActionResult, source, payload), nil
}

// NewThoughtEvent returns the event of the model's thinking, of the source,
// kept whole (its thinking and signature, or its redacted thinking), to be
// given back as it came.
func NewThoughtEvent(source EventSource, chunk LLMChunk) Event {
	thinking := LLMChunk{
		Thinking:         chunk.Thinking,
		Signature:        chunk.Signature,
		RedactedThinking: chunk.RedactedThinking,
	}
	payload, _ := json.Marshal(thinking) // strings marshal

	return NewEvent(EventTypeThought, source, payload)
}

// Text is the payload as text.
func (e Event) Text() string { return string(e.Payload) }

// HasSourceType tells whether the event comes from a source of the type.
func (e Event) HasSourceType(t EventSourceType) bool {
	return e.Source.Type == t
}
