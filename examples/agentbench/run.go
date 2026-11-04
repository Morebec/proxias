package main

import (
	"context"
	"encoding/json"
	"iter"
	"slices"
	"time"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/examples/agentdemo"
)

// run is a run of a scenario on a strategy: what it builds its strategies with,
// and what it times.
type run struct {
	env      agentdemo.Env
	kind     string
	onClaude bool
	t        *timing
}

// strategy is the strategy of the run, on the script or on Claude.
func (r *run) strategy(script agentdemo.Script) proxias.CognitiveStrategy {
	return timedStages(r.env.StrategyOf(r.kind, r.onClaude, script), r.t)
}

// timed is the agent with its turns timed (a middleware of the bench), and its
// executions when its strategy is gated.
func (r *run) timed(a proxias.Agent, strategy proxias.CognitiveStrategy) proxias.Agent {
	a = timedExecution(a, strategy, r.t)
	timer := func(next proxias.CognitiveStrategy) proxias.CognitiveStrategy {
		return proxias.StrategyFunc(
			func(ctx context.Context, inv proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
				return func(yield func(proxias.StreamUpdate) bool) {
					began := time.Now()
					defer func() { r.t.cognition(time.Since(began)) }()
					for u := range next.Invoke(ctx, inv) {
						if !yield(u) {
							return
						}
					}
				}
			},
		)
	}
	a.Middleware = append(slices.Clip(a.Middleware), timer)
	return a
}

// observe observes the stream to its end, and returns the last session, the
// events added, and its error.
func (r *run) observe(stream *proxias.SessionStream) (proxias.Session, []proxias.Event, error) {
	var (
		events []proxias.Event
		first  error
	)
	for d := range stream.Updates() {
		for _, e := range d.Events {
			r.t.observe(e)
		}
		events = append(events, d.Events...)
		if d.Err != nil && first == nil {
			first = d.Err
		}
	}
	return stream.Session(), events, first
}

// loop has the agent Loop on the session.
func (r *run) loop(
	ctx context.Context,
	a proxias.Agent,
	s proxias.Session,
	strategy proxias.CognitiveStrategy,
) (proxias.Session, []proxias.Event, error) {
	timed := r.timed(a, strategy)
	stream, err := timed.Loop(ctx, s, strategy)
	if err != nil {
		return s, nil, err
	}
	return r.observe(stream)
}

// chat has john say the texts to the agent, in one session, a Loop each, and
// returns the reply to the last.
func (r *run) chat(
	ctx context.Context,
	a proxias.Agent,
	strategy proxias.CognitiveStrategy,
	texts ...string,
) (string, error) {
	s := proxias.NewSession()
	var reply string
	for _, text := range texts {
		var (
			events []proxias.Event
			err    error
		)
		john := proxias.NewUserSource("john")
		s = s.WithEvents(proxias.NewMessageEvent(john, text))
		if s, events, err = r.loop(ctx, a, s, strategy); err != nil {
			return "", err
		}
		reply = agentdemo.Reply(events)
	}
	return reply, nil
}

// watch has the agent Watch the events, given in order, unattended, and returns
// the events of the session.
func (r *run) watch(
	ctx context.Context,
	a proxias.Agent,
	strategy proxias.CognitiveStrategy,
	events ...proxias.Event,
) ([]proxias.Event, error) {
	incoming := make(chan proxias.Event, len(events))
	for _, e := range events {
		incoming <- e
	}
	close(incoming)
	timed := r.timed(a, strategy)
	stream, err := timed.Watch(ctx, proxias.NewSession(), incoming, strategy)
	if err != nil {
		return nil, err
	}
	s, _, err := r.observe(stream)
	return s.Events(), err
}

// results are the results of the action in the events.
func results(events []proxias.Event, action string) []proxias.ActionExecutionResult {
	var found []proxias.ActionExecutionResult
	for _, e := range events {
		if e.Type != proxias.EventTypeActionResult {
			continue
		}
		var result proxias.ActionExecutionResult
		err := json.Unmarshal(e.Payload, &result)
		if err == nil && result.ActionID == action {
			found = append(found, result)
		}
	}
	return found
}

// resultsAt tells, for each result of the action in the session, the index of
// the given event it followed.
func resultsAt(session []proxias.Event, given []proxias.Event, action string) []int {
	index := map[proxias.EventID]int{}
	for i, e := range given {
		index[e.ID] = i
	}
	var at []int
	last := -1
	for _, e := range session {
		if i, ok := index[e.ID]; ok {
			last = i
		}
		if len(results([]proxias.Event{e}, action)) > 0 {
			at = append(at, last)
		}
	}
	return at
}
