package agentsdk

import (
	"context"
	"fmt"
)

// DefaultMaxTurns is the turns Loop takes at most when it is not told (WithMaxTurns).
const DefaultMaxTurns = 8

// DefaultStreamBufferSize is the updates a SessionStream holds that its reader has not read yet: the agent's turn
// goes on while they wait, and blocks once they are this many.
const DefaultStreamBufferSize = 16

// AgentBias is a bias the agent reasons with, in words ("prefers asking before acting").
type AgentBias string

// AgentPersona is who the agent is.
type AgentPersona struct {
	Name  string
	Role  string
	Goals []string
}

// Agent is an agent: who it is, what it can do (its execution layer, ActionExecutor), and what wraps every turn it
// takes (AgentMiddleware). How it thinks is the strategy it is invoked with. It holds no session: the app keeps the
// session, so one Agent serves any number of them, concurrently.
type Agent struct {
	// ID identifies the agent: its events come from NewAgentSource(ID).
	ID string
	AgentPersona
	Biases []AgentBias
	// ActionExecutor is the agent's execution layer, nil for no action.
	ActionExecutor ActionExecutor
	// Middleware wrap every turn of the agent, whatever the strategy, the first outermost.
	Middleware []AgentMiddleware
}

// AsSource returns an EventSource representing this agent.
func (a *Agent) AsSource() EventSource { return NewAgentSource(a.ID) }

// Invoke has the agent take one turn, with the strategy, on what happened last in the session (the caller appended
// it), which seeds the stream. It fails at once, without a stream, for no strategy.
func (a *Agent) Invoke(ctx context.Context, s Session, strategy CognitiveStrategy) (*SessionStream, error) {
	if strategy == nil {
		return nil, fmt.Errorf("strategy cannot be nil")
	}
	strategy = a.strategyWithMiddleware(strategy)

	stream := &SessionStream{session: s, updates: make(chan StreamUpdate, DefaultStreamBufferSize)}
	go func() {
		defer stream.close()
		for update := range strategy.Invoke(ctx, Invocation{Agent: *a, Session: stream.Session()}) {
			if err := stream.send(ctx, update); err != nil || update.Err != nil {
				return
			}
		}
	}()
	return stream, nil
}

func (a *Agent) strategyWithMiddleware(strategy CognitiveStrategy) CognitiveStrategy {
	for i := len(a.Middleware) - 1; i >= 0; i-- {
		strategy = a.Middleware[i](strategy)
	}

	return strategy
}

// Loop has the agent take turns (Invoke), with the strategy, until its strategy
// says the agent is done (StopReasonEndTurn). A turn without a StopReason has
// the agent take another, to reason on what it added, until it took its turns:
// then its last update says StopReasonMaxTurns,
// which is not an error. It takes no new turn once ctx ends. It fails at once,
// without a stream, for no strategy.
func (a *Agent) Loop(ctx context.Context, s Session, strategy CognitiveStrategy) (*SessionStream, error) {
	if strategy == nil {
		return nil, fmt.Errorf("strategy cannot be nil")
	}

	stream := &SessionStream{session: s, updates: make(chan StreamUpdate, DefaultStreamBufferSize)}
	go func() {
		defer stream.close()
		for range DefaultMaxTurns {
			if ctx.Err() != nil {
				return
			}
			turn, err := a.Invoke(ctx, stream.Session(), strategy)
			if err != nil {
				_ = stream.send(ctx, StreamUpdate{Err: err})
				return
			}
			for update := range turn.Updates() {
				if err := stream.send(ctx, update); err != nil || update.Err != nil {
					return
				}
			}
			if turn.StopReason() == StopReasonEndTurn {
				return
			}
		}
		_ = stream.send(ctx, StreamUpdate{StopReason: StopReasonMaxTurns}) // ctx ended: nobody to tell
	}()
	return stream, nil
}

// Watch has the agent Loop on each event of events, one at a time: the event is
// appended to the session, then a Loop runs. It closes when events is closed or
// ctx ends; an error of a loop is streamed, and the next event still comes. The
// stream's StopReason is that of the last event's loop. It fails at once,
// without a stream, for no strategy or no events.
func (a *Agent) Watch(ctx context.Context, s Session, events <-chan Event, strategy CognitiveStrategy) (*SessionStream, error) {
	if strategy == nil {
		return nil, fmt.Errorf("strategy cannot be nil")
	}
	if events == nil {
		return nil, fmt.Errorf("events cannot be nil")
	}

	stream := &SessionStream{session: s, updates: make(chan StreamUpdate, DefaultStreamBufferSize)}
	go func() {
		defer stream.close()
		for {
			var (
				e  Event
				ok bool
			)
			select {
			case <-ctx.Done():
				return
			case e, ok = <-events:
				if !ok {
					return
				}
			}
			if err := stream.send(ctx, StreamUpdate{Events: []Event{e}}); err != nil {
				return
			}

			// A Loop on the event: its error is streamed, and the next event still comes.
			loop, err := a.Loop(ctx, stream.Session(), strategy)
			if err != nil {
				if err := stream.send(ctx, StreamUpdate{Err: err}); err != nil {
					return
				}
				continue
			}
			for update := range loop.Updates() {
				if err := stream.send(ctx, update); err != nil {
					return
				}
			}
		}
	}()
	return stream, nil
}
