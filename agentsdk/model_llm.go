package agentsdk

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
)

// Role is who says a message of a conversation with a model.
type Role string

const (
	// RoleUser is everything that happened to the agent.
	RoleUser Role = "user"
	// RoleAssistant is the agent.
	RoleAssistant Role = "assistant"
)

// LLM is a language model, as every strategy sees it: given a system prompt, a conversation and tools, it streams
// text and tool calls. A vendor adapter implements it (anthropic.NewLLM); a script implements it for tests.
type LLM interface {
	Complete(ctx context.Context, req LLMRequest) iter.Seq2[LLMChunk, error]
}

// LLMRequest is what an LLM is asked.
type LLMRequest struct {
	// System is the system prompt, in blocks: the stable ones first, so that a model can cache them.
	System []string

	// Messages is the conversation, oldest first.
	Messages []LLMMessage

	// Tools are the actions the model may call, with their JSON Schemas.
	Tools []ActionDescriptor
}

// LLMMessage is a message of a conversation with a model.
type LLMMessage struct {
	Role  Role
	Parts []LLMPart
}

// LLMPart is a part of a message: text, a tool call, a tool's result, or the model's thinking (Thinking and its
// Signature, or RedactedThinking), given back as the model gave it.
type LLMPart struct {
	Text             string
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
}

// LLMChunk is a piece of what an LLM streams: a piece of text, a whole tool call, or a whole block of thinking
// (Thinking and its Signature, or RedactedThinking), which is to be given back as it came.
type LLMChunk struct {
	Text             string
	ToolCall         *LLMToolCall
	Thinking         string
	Signature        string
	RedactedThinking string
}

// NewLLMRequestFromSession is the session as a request of the model of the agent self: its leading messages of a
// system source are the system prompt, then each event is a message (NewLLMMessage).
func NewLLMRequestFromSession(s Session, self EventSource) LLMRequest {
	var req LLMRequest
	leading := true
	for _, e := range s.Events() {
		if leading = leading && e.Source.Type == EventSourceTypeSystem && e.Type == EventTypeMessage; leading {
			req.System = append(req.System, e.Text())
		} else if m, ok := NewLLMMessage(self, e); ok {
			req.Messages = append(req.Messages, m)
		}
	}
	return req
}

// NewLLMMessage is the event as a message of a conversation with the model of the agent self: self's action requests
// are its tool calls, its thinking (signed) is given back as it came, its other events are its text; the results of
// actions are tool results, a person's messages text, anything else text tagged with its type and source. ok is false
// for an event the model is not told: one of self without text.
func NewLLMMessage(self EventSource, e Event) (m LLMMessage, ok bool) {
	var (
		request  ActionExecutionRequest
		thinking LLMChunk
		tagged   = fmt.Sprintf("[%s from %s] %s", e.Type, e.Source, e.Text())
	)
	switch {
	case e.Source == self && e.Type == EventTypeActionRequest && json.Unmarshal(e.Payload, &request) == nil:
		return LLMMessage{Role: RoleAssistant, Parts: []LLMPart{{ToolCall: &LLMToolCall{ID: string(e.ID), Name: request.ActionID, Input: request.Params}}}}, true
	case e.Source == self && e.Type == EventTypeThought && json.Unmarshal(e.Payload, &thinking) == nil && (thinking.Signature != "" || thinking.RedactedThinking != ""):
		return LLMMessage{Role: RoleAssistant, Parts: []LLMPart{{Thinking: thinking.Thinking, Signature: thinking.Signature, RedactedThinking: thinking.RedactedThinking}}}, true
	case e.Source == self && e.Type == EventTypeActionRequest:
		return LLMMessage{Role: RoleAssistant, Parts: []LLMPart{{Text: tagged}}}, true
	case e.Source == self:
		return LLMMessage{Role: RoleAssistant, Parts: []LLMPart{{Text: e.Text()}}}, len(e.Payload) > 0
	case e.Type == EventTypeActionResult && e.CausationID != "":
		return LLMMessage{Role: RoleUser, Parts: []LLMPart{{ToolResult: &LLMToolResult{CallID: string(e.CausationID), Content: e.Text()}}}}, true
	case e.Source.Type == EventSourceTypeUser && e.Type == EventTypeMessage:
		return LLMMessage{Role: RoleUser, Parts: []LLMPart{{Text: e.Text()}}}, true
	default:
		return LLMMessage{Role: RoleUser, Parts: []LLMPart{{Text: tagged}}}, len(e.Payload) > 0
	}
}
