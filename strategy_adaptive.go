package proxias

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
)

// AdaptiveCognitiveStrategy is the adaptive strategy: a selector chooses, for
// each task, the strategy that suits it among named candidates, and the task
// is that strategy's.
//
// A turn has two phases:
//   - choose: when the turn follows an outside event (a person's message, a
//     probe, a system's message, an event of another agent: anything but the
//     agent's own events and the results of its actions), the selector
//     chooses a candidate from the session, and the choice is yielded as a
//     thought of the agent ({"strategy":"rewoo"}). The turns that follow the
//     agent's own events, until the next outside event, keep to the strategy
//     the session records: a task is one strategy's from start to end.
//   - delegate: the turn is the chosen strategy's.
//
// So it costs the selector's call once a task, on top of the chosen strategy's
// own calls: next to nothing with a System One selector, one small model call
// with an LLM one. It suits agents that meet tasks of different shapes (a
// question to answer, a lookup chained to an email, a task that must adapt),
// each handled by the strategy that fits it. Its trade-off is that a wrong
// choice holds for the whole task; when the selector chooses none, the task is
// the default strategy's.
type AdaptiveCognitiveStrategy struct {
	Selector   StrategySelector
	Candidates []StrategyCandidate
	// Default is the name of the candidate of a task the selector chooses no
	// candidate for: the first candidate when empty.
	Default string
}

// StrategyCandidate is a strategy an AdaptiveCognitiveStrategy can choose: its
// name, what it suits (which the selector chooses by), and the strategy.
type StrategyCandidate struct {
	Name        string
	Description string
	Strategy    CognitiveStrategy
}

// NewAdaptiveStrategy returns the adaptive strategy of the selector and the
// candidates, the first one its default.
func NewAdaptiveStrategy(selector StrategySelector, candidates ...StrategyCandidate) AdaptiveCognitiveStrategy {
	return AdaptiveCognitiveStrategy{
		Selector:   selector,
		Candidates: candidates,
	}
}

// adaptiveChoice is the thought of the agent that records the strategy of a
// task.
type adaptiveChoice struct {
	Strategy string `json:"strategy"`
}

// Invoke takes one turn of the task: the strategy the session records for it,
// or, after an outside event, the one the selector chooses.
func (s AdaptiveCognitiveStrategy) Invoke(ctx context.Context, inv Invocation) iter.Seq[StreamUpdate] {
	return func(yield func(StreamUpdate) bool) {
		name := s.recordedStrategy(inv)
		if name == "" {
			chosen, ok := s.chooseStrategy(ctx, inv, yield)
			if !ok {
				return
			}
			name = chosen
		}

		s.delegate(ctx, inv, name, yield)
	}
}

// recordedStrategy is the name of the strategy the session records for the
// task the turn is in: none when the turn follows an outside event.
func (s AdaptiveCognitiveStrategy) recordedStrategy(inv Invocation) string {
	currentAgent := inv.Agent.AsSource()
	events := inv.Session.Events()
	for i := len(events) - 1; i >= 0; i-- {
		e := events[i]
		if e.Type == EventTypeActionResult {
			continue
		}
		if e.Source != currentAgent {
			return "" // an outside event: a new task
		}

		if e.Type != EventTypeThought {
			continue
		}
		var choice adaptiveChoice
		err := json.Unmarshal(e.Payload, &choice)
		if err != nil {
			continue
		}
		if choice.Strategy != "" {
			return choice.Strategy
		}
	}
	return ""
}

// chooseStrategy has the selector choose the strategy of the task, the default
// when it chooses none, and yields the choice. It returns false when the turn
// ended (an error, yielded, or a stopped stream).
func (s AdaptiveCognitiveStrategy) chooseStrategy(
	ctx context.Context,
	inv Invocation,
	yield func(StreamUpdate) bool,
) (string, bool) {
	name, err := s.Selector.Select(ctx, inv, s.Candidates)
	if err != nil {
		yield(StreamUpdate{Err: err})
		return "", false
	}
	if s.candidate(name) == nil {
		name = s.defaultName()
	}

	payload, _ := json.Marshal(adaptiveChoice{Strategy: name}) // a string marshals
	choice := NewEvent(EventTypeThought, inv.Agent.AsSource(), payload)
	return name, yield(StreamUpdate{Events: []Event{choice}})
}

// delegate has the named strategy take the turn. The choice the turn yielded is
// not in the session the strategy is given: it is the agent's, not the task's.
func (s AdaptiveCognitiveStrategy) delegate(
	ctx context.Context,
	inv Invocation,
	name string,
	yield func(StreamUpdate) bool,
) {
	candidate := s.candidate(name)
	if candidate == nil {
		candidate = s.candidate(s.defaultName())
	}
	if candidate == nil {
		yield(StreamUpdate{Err: errors.New("the adaptive strategy has no candidate")})
		return
	}

	for u := range candidate.Strategy.Invoke(ctx, inv) {
		if !yield(u) || u.Err != nil {
			return
		}
	}
}

// candidate is the candidate of the name: nil when there is none.
func (s AdaptiveCognitiveStrategy) candidate(name string) *StrategyCandidate {
	for i := range s.Candidates {
		if s.Candidates[i].Name == name {
			return &s.Candidates[i]
		}
	}
	return nil
}

// defaultName is the name of the default candidate: Default, or the first.
func (s AdaptiveCognitiveStrategy) defaultName() string {
	if s.Default != "" {
		return s.Default
	}
	if len(s.Candidates) == 0 {
		return ""
	}
	return s.Candidates[0].Name
}
