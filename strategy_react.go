package proxias

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/google/uuid"
)

// ReActCognitiveStrategy is the ReAct strategy (Reason and Act): the model
// thinks and acts in turns, deciding each step after it sees the last one's
// result.
//
// A turn has two phases:
//   - reason: one model call, given who the agent is, the session, and every
//     action of the agent as a tool with its schema; its text is streamed.
//   - act: when the model called tools, the agent runs them, all in one
//     execution, and the turn ends with their results, for the next turn to
//     reason on. When it called none, its answer is the reply.
//
// So a task of N dependent steps takes about N + 1 model calls, each reading
// the whole conversation again. It suits open-ended tasks and conversations,
// where the next step depends on what the last one found. Its trade-off is
// the cost of that adaptivity: ReWOO, which plans the steps up front, takes two
// calls whatever their number, but never sees a result before the answer.
type ReActCognitiveStrategy struct {
	LLM          LLM
	SystemPrompt string
}

// NewReActStrategy returns the ReAct strategy of the model: a turn is one call,
// given who the agent is, the session as a conversation, and every action of
// the agent as a tool with its schema. When the model calls tools, the turn has
// the agent's execution layer run them, and ends with their results, for the
// next turn to reason on (no StopReason: the agent goes on); otherwise it ends
// with the agent's reply (StopReasonEndTurn). The text is streamed as it comes.
func NewReActStrategy(llm LLM) ReActCognitiveStrategy {
	return ReActCognitiveStrategy{LLM: llm}
}

// Invoke takes one ReAct turn: the model reasons, then the agent acts on the
// tools it called.
func (s ReActCognitiveStrategy) Invoke(ctx context.Context, inv Invocation) iter.Seq[StreamUpdate] {
	return func(yield func(StreamUpdate) bool) {
		actionExecRequests, events := s.reason(ctx, inv, yield)
		if len(actionExecRequests) == 0 {
			return
		}

		s.act(ctx, inv, actionExecRequests, events, yield)
	}
}

// reason asks the model, given who the agent is, what it remembers, the
// session, and the agent's actions as tools, and streams its text. When it
// replies, the reply ends the turn. When it calls tools, reason yields its
// thoughts and the requests, and returns the requests and the events it
// yielded, ending with the requests'. It returns no request when the turn ended.
func (s ReActCognitiveStrategy) reason(
	ctx context.Context,
	inv Invocation,
	yield func(StreamUpdate) bool,
) ([]ActionExecutionRequest, []Event) {
	currentAgent := inv.Agent.AsSource()
	sysPrompt := s.SystemPrompt
	if sysPrompt == "" {
		sysPrompt = DefaultReActPrompt
	}

	agentPrompt, err := NewAgentPrompt(ctx, inv)
	if err != nil {
		yield(StreamUpdate{Err: err})
		return nil, nil
	}

	systemInstructions := LLMMessage{
		Role: LLMMessageRoleSystem,
		Parts: []LLMPart{
			{Text: sysPrompt},
			{Text: agentPrompt.String()},
		},
	}

	llmRequest := NewLLMRequestFromSession(currentAgent, inv.Session)
	llmRequest.Messages = slices.Insert(llmRequest.Messages, 0, systemInstructions)
	if inv.Agent.ActionExecutor != nil {
		descriptors, err := inv.Agent.ActionExecutor.Descriptors(ctx, inv.Session)
		if err != nil {
			yield(StreamUpdate{Err: err})
			return nil, nil
		}
		llmRequest.Tools = descriptors
	}

	response := s.LLM.Complete(ctx, llmRequest)
	return s.handleResponse(currentAgent, response, yield)
}

// handleResponse streams the model's answer, its text as it comes, gathers it
// into one message, and yields the events it is (NewEventsFromLLMMessage),
// with the model's stop reason as it is. It returns the requests and the
// events, ending with the requests', or no request when the turn ended.
func (s ReActCognitiveStrategy) handleResponse(
	currentAgent EventSource,
	response iter.Seq2[LLMChunk, error],
	yield func(StreamUpdate) bool,
) ([]ActionExecutionRequest, []Event) {
	// The text is streamed as chunks of the message's event.
	message := LLMMessage{
		ID:   EventID(uuid.NewString()),
		Role: LLMMessageRoleAssistant,
	}
	var (
		requests []ActionExecutionRequest
		text     strings.Builder
		stop     StopReason
	)
	for chunk, err := range response {
		switch {
		case err != nil:
			yield(StreamUpdate{Err: err})
			return nil, nil

		case chunk.StopReason != "":
			stop = chunk.StopReason

		case chunk.isThinking():
			thinking := LLMPart{
				Thinking:         chunk.Thinking,
				Signature:        chunk.Signature,
				RedactedThinking: chunk.RedactedThinking,
			}
			message.Parts = append(message.Parts, thinking)

		case chunk.ToolCall != nil:
			message.Parts = append(message.Parts, LLMPart{ToolCall: chunk.ToolCall})
			request := ActionExecutionRequest{
				ActionID: chunk.ToolCall.Name,
				Params:   chunk.ToolCall.Input,
			}
			requests = append(requests, request)

		case chunk.Text != "":
			text.WriteString(chunk.Text)
			update := StreamUpdate{
				Chunk: &StreamChunk{
					EventID: message.ID,
					Data:    []byte(chunk.Text),
				},
			}
			if !yield(update) {
				return nil, nil
			}
		}
	}
	reply := strings.TrimSpace(text.String())
	message.Parts = append(message.Parts, LLMPart{Text: reply})

	events := NewEventsFromLLMMessage(currentAgent, message)
	if !yield(StreamUpdate{Events: events, StopReason: stop}) {
		return nil, nil
	}

	return requests, events
}

// act has the agent execute the requests, all in one call, the session ending
// with the events of the turn, and yields their results: the turn ends with
// them, for the next to reason on.
func (s ReActCognitiveStrategy) act(
	ctx context.Context,
	inv Invocation,
	requests []ActionExecutionRequest,
	events []Event,
	yield func(StreamUpdate) bool,
) {
	if inv.Agent.ActionExecutor == nil {
		yield(StreamUpdate{Err: errors.New("the model called tools, but the agent has no actions")})
		return
	}

	results, err := inv.Agent.ActionExecutor.Execute(ctx, inv.Session.WithEvents(events...), requests)
	if err == nil && len(results) != len(requests) {
		err = fmt.Errorf("%d results for %d actions", len(results), len(requests))
	}
	if err != nil {
		yield(StreamUpdate{Err: fmt.Errorf("executing actions: %w", err)})
		return
	}

	// The events of the turn end with the requests', in their order.
	requestEvents := events[len(events)-len(requests):]
	resultEvents := make([]Event, len(results))
	for i, result := range results {
		resultEvent, err := NewActionExecutionResultEvent(NewSystemSource("actions"), result)
		if err != nil {
			yield(StreamUpdate{Err: err})
			return
		}
		resultEvent.CausationID = requestEvents[i].ID
		resultEvents[i] = resultEvent
	}
	yield(StreamUpdate{Events: resultEvents})
}
