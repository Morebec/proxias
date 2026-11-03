package agentsdk

import (
	"slices"

	"github.com/google/uuid"
)

// Session is a trace of the agent's experience, like the conversation of a chat completions API: the app keeps it
// and passes it in, and an agent has as many sessions as it has conversations. It is immutable: WithEvents returns a
// new one, and Events a copy, so sessions derived from the same one never share their events.
type Session struct {
	id     string
	events []Event
}

// NewSession returns a session of a new ID, with the events.
func NewSession(events ...Event) Session { return NewSessionWithID(uuid.NewString(), events...) }

// NewSessionWithID returns the session id, with the events.
func NewSessionWithID(id string, events ...Event) Session {
	return Session{id: id}.WithEvents(events...)
}

// ID is the ID of the session.
func (s Session) ID() string { return s.id }

// Events are the events of the session, oldest first: a copy.
func (s Session) Events() []Event { return slices.Clone(s.events) }

// WithEvents returns a new session, of the events of this one then the events, each chained to the one before it (an
// event that has a ParentID keeps it). This session is left as it was: the new one has its own copy of the events.
func (s Session) WithEvents(events ...Event) Session {
	appended := make([]Event, len(s.events), len(s.events)+len(events))
	copy(appended, s.events)
	for _, e := range events {
		if e.ParentID == "" && len(appended) > 0 {
			e.ParentID = appended[len(appended)-1].ID
		}
		appended = append(appended, e)
	}
	return Session{id: s.id, events: appended}
}

// LastEvent is the last event of the session, the zero Event when it has none.
func (s Session) LastEvent() Event {
	if len(s.events) == 0 {
		return Event{}
	}
	return s.events[len(s.events)-1]
}
