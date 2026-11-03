package agentsdk

import (
	"time"

	"github.com/google/uuid"
)

// EventID identifies an event.
type EventID string

// EventType is what an event is to the agent: a message, a thought, an action it requests, the result of one...
type EventType string

const (
	// EventTypeMessage represents a message sent by a given source.
	EventTypeMessage EventType = "message"

	// EventTypeThought represents a thought emitted by an event source (mostly agents for chain of thought).
	EventTypeThought EventType = "cognition.thought"

	// EventTypeActionRequest is an action the agent asks its execution layer for. Its payload is an
	// ActionExecutionRequest, as JSON.
	EventTypeActionRequest EventType = "action.request"

	// EventTypeActionResult is the result of an action request, its CausationID the ID of the request. Its payload is
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
	// EventSourceTypeSystem is anything else: an action runner, a clock, a probe...
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
func NewUserSource(id string) EventSource { return EventSource{Type: EventSourceTypeUser, ID: id} }

// NewAgentSource is the agent id.
func NewAgentSource(id string) EventSource { return EventSource{Type: EventSourceTypeAgent, ID: id} }

// NewSystemSource is the system id.
func NewSystemSource(id string) EventSource { return EventSource{Type: EventSourceTypeSystem, ID: id} }

// String is the source as "type:id", then ":metadata" when there is some.
func (s EventSource) String() string {
	str := string(s.Type) + ":" + s.ID
	if s.Metadata != "" {
		str = str + ":" + s.Metadata
	}
	return str
}

// Event is an entry of the agent's experience: something it perceived, thought, intended or said.
// Events are chained with ParentID, the event they follow, forming a reasoning trace.
type Event struct {
	ID EventID
	// ParentID is the event this one follows in the session, "" for the first.
	ParentID EventID
	// CausationID is the event that caused this one: the action request of an action result.
	CausationID EventID
	Type        EventType
	Source      EventSource
	Payload     []byte
	Date        time.Time
}

// NewEvent returns an event of a new ID, dated now.
func NewEvent(t EventType, source EventSource, payload []byte) Event {
	return Event{ID: EventID(uuid.NewString()), Type: t, Source: source, Payload: payload, Date: time.Now()}
}

// NewMessageEvent returns a message of the source.
func NewMessageEvent(source EventSource, text string) Event {
	return NewEvent(EventTypeMessage, source, []byte(text))
}

// Text is the payload as text.
func (e Event) Text() string { return string(e.Payload) }
