package main

import (
	"context"
	"encoding/json"
	"iter"
	"slices"
	"time"

	"github.com/morebec/proxias/agentsdk"
	"github.com/morebec/proxias/examples/agentdemo"
)

// run is a run of a scenario on a strategy: what it builds its strategies with, and what it times.
type run struct {
	env      agentdemo.Env
	kind     string
	onClaude bool
	t        *timing
}

// strategy is the strategy of the run, on the script or on Claude.
func (r *run) strategy(script agentdemo.Script) agentsdk.CognitiveStrategy {
	return r.env.StrategyOf(r.kind, r.onClaude, script)
}

// timed is the agent with its turns timed: a middleware of the bench.
func (r *run) timed(a agentsdk.Agent) agentsdk.Agent {
	timer := func(next agentsdk.CognitiveStrategy) agentsdk.CognitiveStrategy {
		return agentsdk.StrategyFunc(func(ctx context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
			return func(yield func(agentsdk.StreamUpdate) bool) {
				began := time.Now()
				defer func() { r.t.cognition(time.Since(began)) }()
				for u := range next.Invoke(ctx, inv) {
					if !yield(u) {
						return
					}
				}
			}
		})
	}
	a.Middleware = append(slices.Clip(a.Middleware), timer)
	return a
}

// observe observes the stream to its end, and returns the last session, the events added, and its error.
func (r *run) observe(stream *agentsdk.SessionStream) (agentsdk.Session, []agentsdk.Event, error) {
	var (
		events []agentsdk.Event
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
func (r *run) loop(ctx context.Context, a agentsdk.Agent, s agentsdk.Session, strategy agentsdk.CognitiveStrategy) (agentsdk.Session, []agentsdk.Event, error) {
	timed := r.timed(a)
	stream, err := timed.Loop(ctx, s, strategy)
	if err != nil {
		return s, nil, err
	}
	return r.observe(stream)
}

// chat has john say the texts to the agent, in one session, a Loop each, and returns the reply to the last.
func (r *run) chat(ctx context.Context, a agentsdk.Agent, strategy agentsdk.CognitiveStrategy, texts ...string) (string, error) {
	s := agentsdk.NewSession()
	var reply string
	for _, text := range texts {
		var (
			events []agentsdk.Event
			err    error
		)
		if s, events, err = r.loop(ctx, a, s.WithEvents(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), text)), strategy); err != nil {
			return "", err
		}
		reply = agentdemo.Reply(events)
	}
	return reply, nil
}

// watch has the agent Watch the events, given in order, unattended, and returns the events of the session.
func (r *run) watch(ctx context.Context, a agentsdk.Agent, strategy agentsdk.CognitiveStrategy, events ...agentsdk.Event) ([]agentsdk.Event, error) {
	incoming := make(chan agentsdk.Event, len(events))
	for _, e := range events {
		incoming <- e
	}
	close(incoming)
	timed := r.timed(a)
	stream, err := timed.Watch(ctx, agentsdk.NewSession(), incoming, strategy)
	if err != nil {
		return nil, err
	}
	s, _, err := r.observe(stream)
	return s.Events(), err
}

// results are the results of the action in the events.
func results(events []agentsdk.Event, action string) []agentsdk.ActionExecutionResult {
	var found []agentsdk.ActionExecutionResult
	for _, e := range events {
		var result agentsdk.ActionExecutionResult
		if e.Type == agentsdk.EventTypeActionResult && json.Unmarshal(e.Payload, &result) == nil && result.ActionID == action {
			found = append(found, result)
		}
	}
	return found
}

// resultsAt tells, for each result of the action in the session, the index of the given event it followed.
func resultsAt(session []agentsdk.Event, given []agentsdk.Event, action string) []int {
	index := map[agentsdk.EventID]int{}
	for i, e := range given {
		index[e.ID] = i
	}
	var at []int
	last := -1
	for _, e := range session {
		if i, ok := index[e.ID]; ok {
			last = i
		}
		if len(results([]agentsdk.Event{e}, action)) > 0 {
			at = append(at, last)
		}
	}
	return at
}
