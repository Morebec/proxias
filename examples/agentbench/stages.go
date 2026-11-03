package main

import (
	"context"
	"iter"
	"time"

	"github.com/morebec/proxias/agentsdk"
)

// gatedStage is a stage of a gated turn, as the bench times it.
type gatedStage string

// The stages of a gated turn.
const (
	stageClassify   gatedStage = "classify"
	stageExecute    gatedStage = "execute"
	stageSynthesize gatedStage = "synthesize"
	stageReasoning  gatedStage = "reasoning"
)

// timedClassifier is the classifier, each classification timed.
type timedClassifier struct {
	agentsdk.ActionClassifier
	t *timing
}

// Classify classifies, timed.
func (c timedClassifier) Classify(
	ctx context.Context,
	inv agentsdk.Invocation,
	actions []agentsdk.ActionDescriptor,
) ([]agentsdk.ActionExecutionRequest, error) {
	began := time.Now()
	defer func() { c.t.stage(stageClassify, time.Since(began)) }()
	return c.ActionClassifier.Classify(ctx, inv, actions)
}

// timedExecutor is the agent's actions, each execution timed.
type timedExecutor struct {
	agentsdk.ActionExecutor
	t *timing
}

// Execute executes, timed.
func (e timedExecutor) Execute(
	ctx context.Context,
	s agentsdk.Session,
	requests []agentsdk.ActionExecutionRequest,
) ([]agentsdk.ActionExecutionResult, error) {
	began := time.Now()
	defer func() { e.t.stage(stageExecute, time.Since(began)) }()
	return e.ActionExecutor.Execute(ctx, s, requests)
}

// timedSynthesizer is the synthesizer, each synthesis timed to its last chunk.
type timedSynthesizer struct {
	agentsdk.LLM
	t *timing
}

// Complete completes, timed.
func (l timedSynthesizer) Complete(ctx context.Context, req agentsdk.LLMRequest) iter.Seq2[agentsdk.LLMChunk, error] {
	return func(yield func(agentsdk.LLMChunk, error) bool) {
		began := time.Now()
		defer func() { l.t.stage(stageSynthesize, time.Since(began)) }()
		for c, err := range l.LLM.Complete(ctx, req) {
			if !yield(c, err) {
				return
			}
		}
	}
}

// timedReasoning is the reasoning, each of its turns timed.
func timedReasoning(reasoning agentsdk.CognitiveStrategy, t *timing) agentsdk.CognitiveStrategy {
	return agentsdk.StrategyFunc(
		func(ctx context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
			return func(yield func(agentsdk.StreamUpdate) bool) {
				began := time.Now()
				defer func() { t.stage(stageReasoning, time.Since(began)) }()
				for u := range reasoning.Invoke(ctx, inv) {
					if !yield(u) {
						return
					}
				}
			}
		},
	)
}

// timedStages is the strategy with its stages timed, when it is gated: its
// classifier, its synthesizer and its reasoning decorated.
func timedStages(strategy agentsdk.CognitiveStrategy, t *timing) agentsdk.CognitiveStrategy {
	gated, isGated := strategy.(agentsdk.GatedCognitiveStrategy)
	if !isGated {
		return strategy
	}

	gated.Classifier = timedClassifier{
		ActionClassifier: gated.Classifier,
		t:                t,
	}
	gated.Synthesizer = timedSynthesizer{
		LLM: gated.Synthesizer,
		t:   t,
	}
	gated.Reasoning = timedReasoning(gated.Reasoning, t)
	return gated
}

// timedExecution is the agent with its executions timed, when its strategy is
// gated.
func timedExecution(a agentsdk.Agent, strategy agentsdk.CognitiveStrategy, t *timing) agentsdk.Agent {
	_, isGated := strategy.(agentsdk.GatedCognitiveStrategy)
	if !isGated || a.ActionExecutor == nil {
		return a
	}

	a.ActionExecutor = timedExecutor{
		ActionExecutor: a.ActionExecutor,
		t:              t,
	}
	return a
}
