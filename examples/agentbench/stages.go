package main

import (
	"context"
	"time"

	"github.com/morebec/proxias"
)

// stage is a part of a turn the bench times, summed over a run.
type stage string

// The stages the bench times.
const (
	// stageSelect is the adaptive strategies' choice of a strategy.
	stageSelect stage = "select"
	// stageExecute is the agents' executions of their actions, whatever their
	// strategy.
	stageExecute stage = "execute"
)

// timedSelector is the selector, each choice timed.
type timedSelector struct {
	proxias.StrategySelector
	t *timing
}

// Select selects, timed.
func (s timedSelector) Select(
	ctx context.Context,
	inv proxias.Invocation,
	candidates []proxias.StrategyCandidate,
) (string, error) {
	began := time.Now()
	defer func() { s.t.stage(stageSelect, time.Since(began)) }()
	return s.StrategySelector.Select(ctx, inv, candidates)
}

// timedExecutor is the agent's actions, each execution timed.
type timedExecutor struct {
	proxias.ActionExecutor
	t *timing
}

// Execute executes, timed.
func (e timedExecutor) Execute(
	ctx context.Context,
	s proxias.Session,
	requests []proxias.ActionExecutionRequest,
) ([]proxias.ActionExecutionResult, error) {
	began := time.Now()
	defer func() { e.t.stage(stageExecute, time.Since(began)) }()
	return e.ActionExecutor.Execute(ctx, s, requests)
}

// timedStages is the strategy with its choices timed, when it is adaptive: its
// selector decorated.
func timedStages(strategy proxias.CognitiveStrategy, t *timing) proxias.CognitiveStrategy {
	adaptive, isAdaptive := strategy.(proxias.AdaptiveCognitiveStrategy)
	if !isAdaptive {
		return strategy
	}

	adaptive.Selector = timedSelector{
		StrategySelector: adaptive.Selector,
		t:                t,
	}
	return adaptive
}

// timedExecution is the agent with its executions timed.
func timedExecution(a proxias.Agent, t *timing) proxias.Agent {
	if a.ActionExecutor == nil {
		return a
	}

	a.ActionExecutor = timedExecutor{
		ActionExecutor: a.ActionExecutor,
		t:              t,
	}
	return a
}
