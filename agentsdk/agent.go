package agentsdk

import (
	"context"
	"slices"
	"strconv"
	"time"
)

type AgentBias string

type AgentPersona struct {
	Name  string
	Role  string
	Goals []string
}

type Agent struct {
	AgentPersona
	Biases []AgentBias
	Shell  func(context.Context, Session) ([]Event, error)
}

// Cogitate performs the cogitation of an agent for a certain input event.
// It allows the agent to determine what to do as a result of an event.
// E.g.
//
//		`EventTypeMessage` + `EventSource=user`
//			→ respond conversationally
//		`EventTypeMessage` + `EventSource=agent`
//			→ collaborate or critique
//		`EventTypeThought` + `EventSource=agent`
//			→ continue reasoning
//		`EventTypeActionRequest` + `EventSource=agent`
//			→ wait for result
//		`EventTypeActionResult` + `EventSource=system`
//			→ incorporate observation
//	 `EventTypeMessage` + `EventSource=system`
//			→ treat as signal
func (a Agent) Cogitate(ctx context.Context, e Event) (Session, error) {
	if a.Shell == nil {
		a.Shell = cogitateOnce
	}

	s := NewSession(e)

	for {
		events, err := a.Shell(ctx, s)
		if err != nil {
			// Handle the error (e.g., log it, return a default event, etc.)
			return s, err
		}
		// A shell with nothing more to say ends the loop, rather than spinning on the same session.
		if len(events) == 0 {
			break
		}
		s.Events = append(s.Events, events...)

		if slices.ContainsFunc(events, Event.Terminal) {
			break
		}
	}

	return s, nil
}

func cogitateOnce(ctx context.Context, s Session) ([]Event, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	e := s.LastEvent()

	if e.Terminal() {
		return nil, nil
	}

	turnCount := len(s.Events)
	if turnCount >= 10 {
		return []Event{
			{
				Type:    EventTypeCognitionCompleted,
				Source:  EventSource{Type: EventSourceTypeAgent, ID: "cognition"},
				Payload: []byte("Cognition interrupted after " + strconv.Itoa(turnCount) + " turns."),
				Date:    time.Now(),
			},
		}, nil
	}

	return anthropicComplete(ctx, s)
}
