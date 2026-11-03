package agentsdk

import (
	"context"
	"iter"
)

// CognitiveStrategy is how an agent thinks, one turn at a time for the given
// strategy. Within its turn it may call models and have the agent execute
// actions (Agent.Execute); it asks the agent's ActionExecutor for the
// descriptors, and never narrows them. Its prompts are its own.
type CognitiveStrategy interface {
	Invoke(ctx context.Context, inv Invocation) iter.Seq[StreamUpdate]
}

// Invocation is what a turn is given: the agent, its session, and the mode it
// is invoked in.
type Invocation struct {
	Agent   Agent
	Session Session
	// Memories are what the agent's memory recalled for the turn: not events of
	// the session.
	Memories []Event
}

// LastEvent is the last event of the invocation's session.
func (inv Invocation) LastEvent() Event {
	return inv.Session.LastEvent()
}

// StrategyFunc is a function as a CognitiveStrategy.
type StrategyFunc func(ctx context.Context, inv Invocation) iter.Seq[StreamUpdate]

// Invoke calls f.
func (f StrategyFunc) Invoke(ctx context.Context, inv Invocation) iter.Seq[StreamUpdate] {
	return f(ctx, inv)
}

// AgentMiddleware wraps every turn of an agent, whatever the strategy
// (Agent.Middleware): it may add to the session it is given (an event, a system
// message), change or filter what the turn adds, time it, guard it...
type AgentMiddleware func(next CognitiveStrategy) CognitiveStrategy
