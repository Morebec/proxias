package agentsdk

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/google/uuid"
)

// Session is a trace of the agent's experience, like the conversation of a chat completions API: the app keeps it
// and passes it in, and an agent has as many sessions as it has conversations. It is immutable: WithEvents returns a
// new one, and Events a copy, so sessions derived from the same one never share their events.
type Session struct {
	id     string
	events []Event
}

// NewSession returns a session of a new ID, with the events.
func NewSession(events ...Event) Session { return NewSessionWithID(uuid.NewString(), events...) }

// NewSessionWithID returns the session id, with the events.
func NewSessionWithID(id string, events ...Event) Session {
	return Session{id: id}.WithEvents(events...)
}

// ID is the ID of the session.
func (s Session) ID() string { return s.id }

// Events are the events of the session, oldest first: a copy.
func (s Session) Events() []Event { return slices.Clone(s.events) }

// WithEvents returns a new session, of the events of this one then the events, each chained to the one before it (an
// event that has a ParentID keeps it). This session is left as it was: the new one has its own copy of the events.
func (s Session) WithEvents(events ...Event) Session {
	appended := make([]Event, len(s.events), len(s.events)+len(events))
	copy(appended, s.events)
	for _, e := range events {
		if e.ParentID == "" && len(appended) > 0 {
			e.ParentID = appended[len(appended)-1].ID
		}
		appended = append(appended, e)
	}
	return Session{id: s.id, events: appended}
}

// LastEvent is the last event of the session, the zero Event when it has none.
func (s Session) LastEvent() Event {
	if len(s.events) == 0 {
		return Event{}
	}
	return s.events[len(s.events)-1]
}

// conversation is the session of the agent self as a conversation with a model:
//   - the instructions: the messages of a system source before any other event, to join the system prompt;
//   - the messages: self's thoughts, messages and action requests (tool calls) as RoleAssistant; the
//     results of the actions (tool results), a person's messages, and anything else, tagged with its type and
//     source, as RoleUser. Consecutive messages of one role are one message.
//
// With toolsAsText, tool calls and results are told as tagged text instead, for a model given no tools.
func (s Session) conversation(self EventSource, toolsAsText bool) ([]string, []LLMMessage) {
	var (
		instructions []string
		messages     []LLMMessage
		leading      = true
	)
	add := func(role Role, part LLMPart) {
		if n := len(messages); n > 0 && messages[n-1].Role == role {
			messages[n-1].Parts = append(messages[n-1].Parts, part)
			return
		}
		messages = append(messages, LLMMessage{Role: role, Parts: []LLMPart{part}})
	}
	tagged := func(e Event) string { return fmt.Sprintf("[%s from %s] %s", e.Type, e.Source, e.Text()) }

	for _, e := range s.events {
		if leading && e.Source.Type == EventSourceTypeSystem && e.Type == EventTypeMessage {
			instructions = append(instructions, e.Text())
			continue
		}
		leading = false
		switch {
		case e.Source == self && e.Type == EventTypeActionRequest:
			var intent ActionExecutionRequest
			if toolsAsText || json.Unmarshal(e.Payload, &intent) != nil {
				add(RoleAssistant, LLMPart{Text: tagged(e)})
				continue
			}
			add(RoleAssistant, LLMPart{ToolCall: &LLMToolCall{ID: string(e.ID), Name: intent.ActionID, Input: intent.Params}})
		case e.Source == self && e.Type == EventTypeThought:
			// A thought of the model's own thinking is given back as it came (left out of a conversation told as text);
			// another is text.
			var thinking LLMChunk
			if json.Unmarshal(e.Payload, &thinking) == nil && (thinking.Signature != "" || thinking.RedactedThinking != "") {
				if !toolsAsText {
					add(RoleAssistant, LLMPart{Thinking: thinking.Thinking, Signature: thinking.Signature, RedactedThinking: thinking.RedactedThinking})
				}
				continue
			}
			add(RoleAssistant, LLMPart{Text: e.Text()})
		case e.Source == self:
			if len(e.Payload) > 0 {
				add(RoleAssistant, LLMPart{Text: e.Text()})
			}
		case e.Type == EventTypeActionResult && e.CausationID != "" && !toolsAsText:
			add(RoleUser, LLMPart{ToolResult: &LLMToolResult{CallID: string(e.CausationID), Content: e.Text()}})
		case e.Source.Type == EventSourceTypeUser && e.Type == EventTypeMessage:
			add(RoleUser, LLMPart{Text: e.Text()})
		case len(e.Payload) > 0:
			add(RoleUser, LLMPart{Text: tagged(e)})
		}
	}
	return instructions, messages
}
