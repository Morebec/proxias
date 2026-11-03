package agentsdk

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ReActCognitiveStrategy is the ReAct strategy.
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

// handleResponse streams the model's answer: its text as it comes, its thinking
// kept whole (to be given back as it came), its tool calls as requests. Without
// tool calls, its reply ends the turn. With them, it yields its thoughts and
// the requests, and returns the requests and the events it yielded, ending with
// the requests'. It returns no request when the turn ended. The model's stop
// reason is passed on as it is.
func (s ReActCognitiveStrategy) handleResponse(
	currentAgent EventSource,
	response iter.Seq2[LLMChunk, error],
	yield func(StreamUpdate) bool,
) ([]ActionExecutionRequest, []Event) {
	var (
		thoughts      []Event
		requests      []ActionExecutionRequest
		requestEvents []Event
		text          strings.Builder
		textID        = EventID(uuid.NewString())
		stop          StopReason
	)
	for chunk, err := range response {
		switch {
		case err != nil:
			yield(StreamUpdate{Err: err})
			return nil, nil

		case chunk.StopReason != "":
			stop = chunk.StopReason

		case chunk.isThinking():
			thoughts = append(thoughts, NewThoughtEvent(currentAgent, chunk))

		case chunk.ToolCall != nil:
			request := ActionExecutionRequest{
				ActionID: chunk.ToolCall.Name,
				Params:   chunk.ToolCall.Input,
			}
			requestEvent, err := NewActionExecutionRequestEvent(currentAgent, request)
			if err != nil {
				yield(StreamUpdate{Err: err})
				return nil, nil
			}
			requestEvent.ID = EventID(chunk.ToolCall.ID) // the result names the call
			requests = append(requests, request)
			requestEvents = append(requestEvents, requestEvent)

		case chunk.Text != "":
			text.WriteString(chunk.Text)
			update := StreamUpdate{
				Chunk: &StreamChunk{
					EventID: textID,
					Data:    []byte(chunk.Text),
				},
			}
			if !yield(update) {
				return nil, nil
			}
		}
	}
	reply := Event{
		ID:      textID,
		Type:    EventTypeMessage,
		Source:  currentAgent,
		Payload: []byte(strings.TrimSpace(text.String())),
		Date:    time.Now(),
	}

	// An answer without tool calls is the reply: it ends the turn.
	if len(requests) == 0 {
		yield(StreamUpdate{Events: append(thoughts, reply), StopReason: stop})
		return nil, nil
	}

	// Text with tool calls is what the model thinks.
	events := thoughts
	if len(reply.Payload) > 0 {
		reply.Type = EventTypeThought
		events = append(events, reply)
	}
	events = append(events, requestEvents...)
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
