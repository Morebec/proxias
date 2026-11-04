package proxias

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/google/uuid"
)

// GatedCognitiveStrategy is the gated strategy: the classifier
// (a CompositeActionClassifier, or the app's own) finds the executions the last
// message asks for, the agent executes them, and the synthesizer writes the
// reply from their results, the turn's final output.
// When the classifier finds none (the last message asks for no action), or the
// session's last event is the agent's own or an action result (its reasoning
// going on), or the agent has no actions, the turn is Reasoning's.
type GatedCognitiveStrategy struct {
	Classifier  ActionClassifier
	Synthesizer LLM
	Reasoning   CognitiveStrategy
	// SynthesisPrompt is the system prompt of the synthesis:
	// DefaultSynthesisPrompt when empty.
	SynthesisPrompt string
}

// NewGatedStrategy returns the gated strategy of the classifier, the
// synthesizer and the reasoning.
func NewGatedStrategy(
	classifier ActionClassifier,
	synthesizer LLM,
	reasoning CognitiveStrategy,
) GatedCognitiveStrategy {
	return GatedCognitiveStrategy{
		Classifier:  classifier,
		Synthesizer: synthesizer,
		Reasoning:   reasoning,
	}
}

// Invoke takes one gated turn: classify, execute, synthesize; or reason.
func (g GatedCognitiveStrategy) Invoke(ctx context.Context, inv Invocation) iter.Seq[StreamUpdate] {
	return func(yield func(StreamUpdate) bool) {
		agent := inv.Agent
		lastEvent := inv.LastEvent()

		// An agent without actions has nothing to classify: it reasons.
		if agent.ActionExecutor == nil {
			g.reason(ctx, inv, yield)
			return
		}

		// After its own event, or an action's result, the agent reasons on.
		if lastEvent.Source == agent.AsSource() || lastEvent.Type == EventTypeActionResult {
			g.reason(ctx, inv, yield)
			return
		}

		descriptors, err := agent.ActionExecutor.Descriptors(ctx, inv.Session)
		if err != nil {
			yield(StreamUpdate{Err: err})
			return
		}

		var requests []ActionExecutionRequest
		if len(descriptors) != 0 {
			requests, err = g.Classifier.Classify(ctx, inv, descriptors)
			if err != nil {
				yield(StreamUpdate{Err: err})
				return
			}
		}

		if len(requests) == 0 {
			g.reason(ctx, inv, yield)
			return
		}

		g.executeAndSynthesize(ctx, inv, requests, yield)
	}
}

// reason has the reasoning take the turn.
func (g GatedCognitiveStrategy) reason(ctx context.Context, inv Invocation, yield func(StreamUpdate) bool) {
	for u := range g.Reasoning.Invoke(ctx, inv) {
		if !yield(u) {
			return
		}

		if u.Err != nil {
			return // the reasoning failed: the turn ends with its error
		}
	}
}

// executeAndSynthesize runs the requested actions, and synthesizes their
// results into the reply.
func (g GatedCognitiveStrategy) executeAndSynthesize(
	ctx context.Context,
	inv Invocation,
	requests []ActionExecutionRequest,
	yield func(StreamUpdate) bool,
) {
	executedEvents, err := g.executeActionRequests(ctx, inv, requests)
	if err != nil {
		yield(StreamUpdate{Err: err})
		return
	}
	if !yield(StreamUpdate{Events: executedEvents}) {
		return
	}

	g.synthesize(ctx, inv, executedEvents, yield)
}

// executeActionRequests has the agent execute the requests, all in one call,
// and returns their events, then the events of their results.
func (g GatedCognitiveStrategy) executeActionRequests(
	ctx context.Context,
	inv Invocation,
	requests []ActionExecutionRequest,
) ([]Event, error) {
	agent := inv.Agent
	lastEvent := inv.LastEvent()

	requestEvents := make([]Event, len(requests), 2*len(requests)) // the results are appended to them
	for i, request := range requests {
		e, err := NewActionExecutionRequestEvent(agent.AsSource(), request)
		if err != nil {
			return nil, err
		}
		e.CausationID = lastEvent.ID
		requestEvents[i] = e
	}

	results, err := agent.ActionExecutor.Execute(ctx, inv.Session.WithEvents(requestEvents...), requests)
	if err != nil {
		return nil, fmt.Errorf("executing actions: %w", err)
	}
	if len(results) != len(requests) {
		return nil, fmt.Errorf("%d results for %d actions", len(results), len(requests))
	}

	resultEvents := make([]Event, len(results))
	for i, result := range results {
		e, err := NewActionExecutionResultEvent(NewSystemSource("actions"), result)
		if err != nil {
			return nil, err
		}
		e.CausationID = requestEvents[i].ID
		resultEvents[i] = e
	}

	return append(requestEvents, resultEvents...), nil
}

// synthesize has the synthesizer write the reply from the executed requests
// and their results: the turn's final output.
func (g GatedCognitiveStrategy) synthesize(
	ctx context.Context,
	inv Invocation,
	executedEvents []Event,
	yield func(StreamUpdate) bool,
) {
	llmRequest, err := g.synthesisRequest(ctx, inv, executedEvents)
	if err != nil {
		yield(StreamUpdate{Err: err})
		return
	}

	// The synthesizer's answer, its text streamed as it comes, gathered into
	// one message.
	message := LLMMessage{
		ID:   EventID(uuid.NewString()),
		Role: LLMMessageRoleAssistant,
	}
	var (
		text strings.Builder
		stop StopReason
	)
	for chunk, err := range g.Synthesizer.Complete(ctx, llmRequest) {
		switch {
		case err != nil:
			yield(StreamUpdate{Err: err})
			return
		case chunk.StopReason != "":
			stop = chunk.StopReason
		case chunk.isThinking():
			thinking := LLMPart{
				Thinking:         chunk.Thinking,
				Signature:        chunk.Signature,
				RedactedThinking: chunk.RedactedThinking,
			}
			message.Parts = append(message.Parts, thinking)
		case chunk.Text != "":
			text.WriteString(chunk.Text)
			update := StreamUpdate{
				Chunk: &StreamChunk{
					EventID: message.ID,
					Data:    []byte(chunk.Text),
				},
			}
			if !yield(update) {
				return
			}
		}
	}
	reply := strings.TrimSpace(text.String())
	message.Parts = append(message.Parts, LLMPart{Text: reply})

	events := NewEventsFromLLMMessage(inv.Agent.AsSource(), message)
	yield(StreamUpdate{Events: events, StopReason: stop})
}

// synthesisRequest is the request of the synthesis: its prompt and who the
// agent is, then the session, ending with the executed requests and their
// results.
func (g GatedCognitiveStrategy) synthesisRequest(
	ctx context.Context,
	inv Invocation,
	executedEvents []Event,
) (LLMRequest, error) {
	synthesisPrompt := g.SynthesisPrompt
	if synthesisPrompt == "" {
		synthesisPrompt = DefaultSynthesisPrompt
	}

	agentPrompt, err := NewAgentPrompt(ctx, inv)
	if err != nil {
		return LLMRequest{}, err
	}

	systemInstructions := LLMMessage{
		Role: LLMMessageRoleSystem,
		Parts: []LLMPart{
			{Text: synthesisPrompt},
			{Text: agentPrompt.String()},
		},
	}

	llmRequest := NewLLMRequestFromSession(inv.Agent.AsSource(), inv.Session.WithEvents(executedEvents...))
	llmRequest.Messages = slices.Insert(llmRequest.Messages, 0, systemInstructions)
	return llmRequest, nil
}
