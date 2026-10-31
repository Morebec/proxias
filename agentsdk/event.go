package agentsdk

import "time"

type EventType string

const (
	// EventTypeMessage represents a message sent by a given source.
	EventTypeMessage EventType = "message"

	// EventTypeThought represents a thought emitted by an event source (mostly agents for chain of thought).
	EventTypeThought EventType = "cognition.thought"

	// EventTypeCognitionCompleted marker indicating that an event represents the end of a cognition loop.
	EventTypeCognitionCompleted EventType = "cognition.completed"

	// EventTypeActionRequest  represents an event corresponding to the execution of a tool.
	EventTypeActionRequest EventType = "action.request"

	EventTypeActionResult EventType = "action.result"
)

const (
	EventSourceTypeUser   = "user"
	EventSourceTypeAgent  = "agent"
	EventSourceTypeSystem = "system"
)

// EventSource indicates from where an event originated.
// This can be a user, an agent, a tool call, a system's state, etc.
type EventSource struct {
	Type     string
	ID       string
	Metadata string
}

func (s EventSource) String() string {
	str := s.Type + ":" + s.ID
	if s.Metadata != "" {
		str = str + ":" + s.Metadata
	}
	return str
}

// Event an event is an occurrence of a thing having happened in the agent's environment.
// An event can represent as much a user message, a response of the agent,
// a thought fed back to self to an agent, a tool call, a system state, etc.
// Events are chained with a pointer to their previous event forming a ** reasoning trace**.
type Event struct {
	Type    EventType
	Source  EventSource
	Payload []byte
	Date    time.Time
}

func (e Event) Terminal() bool { return e.Type == EventTypeCognitionCompleted }

// FormatEventToProtocol writes an event back in the markers of the cognition protocol,
// as the model would have written it. It is the inverse of the parsing of a response:
// without the markers, the model reads its own past thoughts and actions as plain text,
// and believes it never emitted them. Events that the protocol has no marker for are their payload.
func FormatEventToProtocol(e Event) string {
	payload := string(e.Payload)
	switch e.Type {
	case EventTypeThought:
		return thoughtOpenMarker + payload + thoughtCloseMarker
	case EventTypeActionRequest:
		return actionOpenMarker + payload + actionCloseMarker
	case EventTypeCognitionCompleted:
		return payload + doneMarker
	default:
		return payload
	}
}

func FormatEventToXMLish(e Event) string {
	return `<event>
type: ` + string(e.Type) + `
source: ` + e.Source.String() + `
timestamp: ` + e.Date.Format(time.RFC3339) + `
payload:` + string(e.Payload) + ` 
</event>
`
}
