package proxias

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"
)

// LLMMessageRole is who says a message of a conversation with a model.
type LLMMessageRole string

const (
	// LLMMessageRoleSystem is the instructions: the strategy's, and the
	// messages of a system source.
	LLMMessageRoleSystem LLMMessageRole = "system"
	// LLMMessageRoleUser is everything that happened to the agent.
	LLMMessageRoleUser LLMMessageRole = "user"
	// LLMMessageRoleAssistant is the agent.
	LLMMessageRoleAssistant LLMMessageRole = "assistant"
	// LLMMessageRoleTool is the results of the agent's tool calls.
	LLMMessageRoleTool LLMMessageRole = "tool"
)

// LLM is a language model, as every strategy sees it: given a conversation (its
// instructions as system messages) and tools, it streams text and tool calls. A
// vendor adapter
// implements it (anthropic.NewLLM); a script implements it for tests.
type LLM interface {
	Complete(ctx context.Context, req LLMRequest) iter.Seq2[LLMChunk, error]
}

// LLMRequest is what an LLM is asked.
type LLMRequest struct {
	// Messages is the conversation, oldest first. The system messages that open
	// it are the instructions: the stable ones first, so that a model can cache
	// them.
	Messages []LLMMessage

	// Tools are the actions the model may call, with their JSON Schemas.
	Tools []ActionDescriptor

	// Model is the model to ask, overriding the LLM's own when set: a vendor's
	// name for it (claude-haiku-5-5, gpt-5-mini).
	Model string

	// Effort is how much the model spends on its answer, its thinking
	// included, overriding the LLM's own when set.
	Effort Effort
}

// Effort is how much a model spends on an answer, its thinking included: from
// EffortLow, for quick calls (a choice, a classification), to EffortMax, for
// long work. Empty is the model's default.
type Effort string

const (
	// EffortLow is the least a model spends.
	EffortLow Effort = "low"
	// EffortMedium is a balance of cost and depth.
	EffortMedium Effort = "medium"
	// EffortHigh is for hard work.
	EffortHigh Effort = "high"
	// EffortXHigh is more than high, where a model has it.
	EffortXHigh Effort = "xhigh"
	// EffortMax is the most a model spends.
	EffortMax Effort = "max"
)

// LLMMessage is a message of a conversation with a model.
type LLMMessage struct {
	// ID is the ID of the event of the message's text, when it is known: the
	// event its streamed chunks are pieces of.
	ID    EventID `json:",omitempty"`
	Role  LLMMessageRole
	Parts []LLMPart
}

// LLMPart is a part of a message: text, a file, a tool call, a tool's result,
// or the model's thinking (Thinking and its Signature, or RedactedThinking),
// given back as the model gave it.
type LLMPart struct {
	Text             string
	File             *File
	ToolCall         *LLMToolCall
	ToolResult       *LLMToolResult
	Thinking         string
	Signature        string
	RedactedThinking string
}

// LLMToolCall is a model's call of a tool: an action request.
type LLMToolCall struct {
	ID    string
	Name  string
	Input map[string]any
}

// LLMToolResult is the result of a tool call: an action result.
type LLMToolResult struct {
	CallID  string
	Content string
	// Files is the files the action gives the model to read, after Content.
	Files []File
}

// LLMChunk is a piece of what an LLM streams: a piece of text, a piece of
// thinking as the model writes it (ThinkingDelta, only to be shown: the whole
// block follows), a whole tool call, or a whole block of thinking (Thinking and
// its Signature, or RedactedThinking), which is to be given back as it came. When the model
// stopped (it did not ask for tool calls), the last chunk says why
// (StopReason),
// and only that.
type LLMChunk struct {
	Text             string
	ThinkingDelta    string `json:",omitempty"`
	ToolCall         *LLMToolCall
	Thinking         string
	Signature        string
	RedactedThinking string
	StopReason       StopReason `json:",omitempty"`
}

// isThinking tells whether the chunk is a block of thinking, signed or not.
func (c LLMChunk) isThinking() bool {
	return c.Thinking != "" || c.Signature != "" || c.RedactedThinking != ""
}

// NewLLMRequestFromSession creates an LLMRequest from a session.
func NewLLMRequestFromSession(currentAgentSource EventSource, s Session) LLMRequest {
	var req LLMRequest
	events := s.Events()

	for _, e := range events {
		req.Messages = append(req.Messages, NewLLMMessageFromEvent(currentAgentSource, e))
	}

	return req
}

// NewLLMMessageFromEvent creates the message an event is, in the conversation
// of the model of the current agent: its own events are the assistant's, action
// results the tool's, a system's messages (its instructions) the system's, a
// person's messages the user's, and everything else (another agent's events, a
// system's observations) the user's,
// tagged with its type and source.
func NewLLMMessageFromEvent(currentAgentSource EventSource, e Event) LLMMessage {
	switch e.Type {
	case EventTypeActionRequest:
		return newLLMMessageFromActionRequestEvent(currentAgentSource, e)
	case EventTypeActionResult:
		return newLLMMessageFromActionResultEvent(e)
	case EventTypeThought:
		return newLLMMessageFromThoughtEvent(currentAgentSource, e)
	case EventTypeMessage:
		return newLLMMessageFromMessageEvent(currentAgentSource, e)
	default:
		return newLLMMessageFromUnknownEvent(currentAgentSource, e)
	}
}

// newLLMMessageFromActionRequestEvent: the current agent's request is its tool call,
// or its text, tagged, when it is not a request; another's request is the
// user's
// text, tagged.
func newLLMMessageFromActionRequestEvent(currentAgentSource EventSource, e Event) LLMMessage {
	tagged := externalTextPayload(e)
	if e.Source != currentAgentSource {
		return LLMMessage{
			Role: LLMMessageRoleUser,
			Parts: []LLMPart{
				{Text: tagged},
			},
		}
	}

	var request ActionExecutionRequest
	err := json.Unmarshal(e.Payload, &request)
	if err != nil {
		return LLMMessage{
			Role: LLMMessageRoleAssistant,
			Parts: []LLMPart{
				{Text: tagged},
			},
		}
	}

	return LLMMessage{
		Role: LLMMessageRoleAssistant,
		Parts: []LLMPart{
			{
				ToolCall: &LLMToolCall{
					ID:    string(e.ID),
					Name:  request.ActionID,
					Input: request.Params,
				},
			},
		},
	}
}

// newLLMMessageFromThoughtEvent: the current agent's signed thinking is given back
// as
// it came, its other thoughts are its text; another's thought is the user's
// text, tagged.
func newLLMMessageFromThoughtEvent(currentAgentSource EventSource, e Event) LLMMessage {
	if e.Source != currentAgentSource {
		return LLMMessage{
			Role: LLMMessageRoleUser,
			Parts: []LLMPart{
				{Text: externalTextPayload(e)},
			},
		}
	}

	var chunk LLMChunk
	err := json.Unmarshal(e.Payload, &chunk)
	if err != nil || !chunk.isThinking() {
		return LLMMessage{
			Role: LLMMessageRoleAssistant,
			Parts: []LLMPart{
				{Text: e.Text()},
			},
		}
	}

	part := LLMPart{
		Thinking:         chunk.Thinking,
		Signature:        chunk.Signature,
		RedactedThinking: chunk.RedactedThinking,
	}
	return LLMMessage{
		Role:  LLMMessageRoleAssistant,
		Parts: []LLMPart{part},
	}
}

// newLLMMessageFromActionResultEvent: a result is the tool's, the result of the call
// it was caused by.
func newLLMMessageFromActionResultEvent(e Event) LLMMessage {
	return LLMMessage{
		Role: LLMMessageRoleTool,
		Parts: []LLMPart{
			{
				ToolResult: &LLMToolResult{
					CallID:  string(e.CausationID),
					Content: e.Text(),
					Files:   e.Files,
				},
			},
		},
	}
}

// newLLMMessageFromMessageEvent: the current agent's message is its text, a system's
// an instruction, a person's the user's text, another's (an agent's) the user's
// text, tagged.
func newLLMMessageFromMessageEvent(currentAgentSource EventSource, e Event) LLMMessage {
	if e.Source == currentAgentSource {
		return LLMMessage{
			Role: LLMMessageRoleAssistant,
			Parts: []LLMPart{
				{Text: e.Text()},
			},
		}
	}

	if e.HasSourceType(EventSourceTypeSystem) {
		return LLMMessage{
			Role: LLMMessageRoleSystem,
			Parts: []LLMPart{
				{Text: e.Text()},
			},
		}
	}

	if e.HasSourceType(EventSourceTypeUser) {
		return LLMMessage{
			Role:  LLMMessageRoleUser,
			Parts: withFileParts([]LLMPart{{Text: e.Text()}}, e.Files),
		}
	}

	return LLMMessage{
		Role:  LLMMessageRoleUser,
		Parts: withFileParts([]LLMPart{{Text: externalTextPayload(e)}}, e.Files),
	}
}

// withFileParts returns the parts followed by one part per file. Only the
// user's messages hold files: models read files only from the user.
func withFileParts(parts []LLMPart, files []File) []LLMPart {
	for i := range files {
		parts = append(parts, LLMPart{File: &files[i]})
	}
	return parts
}

// newLLMMessageFromUnknownEvent: an event of another type (a system's
// observation, a probe's say, or the app's own) is the user's text, tagged with
// its type and source (the current agent's, its text).
func newLLMMessageFromUnknownEvent(currentAgentSource EventSource, e Event) LLMMessage {
	if e.Source == currentAgentSource {
		return LLMMessage{
			Role: LLMMessageRoleAssistant,
			Parts: []LLMPart{
				{Text: e.Text()},
			},
		}
	}

	return LLMMessage{
		Role: LLMMessageRoleUser,
		Parts: []LLMPart{
			{Text: externalTextPayload(e)},
		},
	}
}

// externalTextPayload is the event's payload, tagged with its type and source, for
// the model of another agent.
func externalTextPayload(e Event) string {
	return fmt.Sprintf("[%s from %s] %s", e.Type, e.Source, e.Text())
}

// NewEventsFromLLMMessage creates the events a message of the model of the
// current agent is, in its session: its thinking as thoughts, then its text,
// then its tool calls as action requests (of the calls' IDs). Its text is a
// message; with tool calls, it is a thought (text with tool calls is what the
// model thinks), left out when empty. The mirror of NewLLMMessageFromEvent.
func NewEventsFromLLMMessage(currentAgentSource EventSource, m LLMMessage) []Event {
	events := newThoughtEventsFromLLMMessage(currentAgentSource, m)
	requests := newActionRequestEventsFromLLMMessage(currentAgentSource, m)
	if len(requests) == 0 {
		return append(events, newMessageEventFromLLMMessage(currentAgentSource, m))
	}

	thought := newMessageEventFromLLMMessage(currentAgentSource, m)
	if len(thought.Payload) > 0 {
		thought.Type = EventTypeThought
		events = append(events, thought)
	}
	return append(events, requests...)
}

// newThoughtEventsFromLLMMessage: the message's thinking, signed or not, is
// its thoughts, kept whole.
func newThoughtEventsFromLLMMessage(currentAgentSource EventSource, m LLMMessage) []Event {
	var thoughts []Event
	for _, part := range m.Parts {
		chunk := LLMChunk{
			Thinking:         part.Thinking,
			Signature:        part.Signature,
			RedactedThinking: part.RedactedThinking,
		}
		if chunk.isThinking() {
			thoughts = append(thoughts, NewThoughtEvent(currentAgentSource, chunk))
		}
	}
	return thoughts
}

// newMessageEventFromLLMMessage: the message's text is a message, of the
// message's ID when it has one, empty when it says nothing.
func newMessageEventFromLLMMessage(currentAgentSource EventSource, m LLMMessage) Event {
	var text strings.Builder
	for _, part := range m.Parts {
		text.WriteString(part.Text)
	}
	e := NewMessageEvent(currentAgentSource, strings.TrimSpace(text.String()))
	if m.ID != "" {
		e.ID = m.ID
	}
	return e
}

// newActionRequestEventsFromLLMMessage: the message's tool calls are action
// requests, of the calls' IDs, which their results name.
func newActionRequestEventsFromLLMMessage(currentAgentSource EventSource, m LLMMessage) []Event {
	var requests []Event
	for _, part := range m.Parts {
		if part.ToolCall == nil {
			continue
		}
		request := ActionExecutionRequest{
			ActionID: part.ToolCall.Name,
			Params:   part.ToolCall.Input,
		}
		e, _ := NewActionExecutionRequestEvent(currentAgentSource, request) // the input of a call was JSON
		e.ID = EventID(part.ToolCall.ID)
		requests = append(requests, e)
	}
	return requests
}
