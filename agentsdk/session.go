package agentsdk

import "github.com/google/uuid"

type Session struct {
	SessionID string
	Events    []Event
}

func NewSession(e Event) Session { return NewSessionWithID(uuid.New().String(), e) }

func NewSessionWithID(id string, e Event) Session {
	return Session{
		SessionID: id,
		Events:    []Event{e},
	}
}

func (s Session) LastEvent() Event {
	if len(s.Events) == 0 {
		return Event{}
	}

	return s.Events[len(s.Events)-1]
}
