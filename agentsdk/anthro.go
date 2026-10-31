package agentsdk

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

const APIKey = "REMOVED-API-KEY"

func anthropicComplete(ctx context.Context, s Session) ([]Event, error) {
	client := anthropic.NewClient(option.WithAPIKey(APIKey))

	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     "claude-haiku-5-5",
		MaxTokens: 4096,
		System: []anthropic.TextBlockParam{{
			Text:         SystemPrompt,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}},
		Messages: mapEventHistoryToEventParams(s.Events),
	})
	if err != nil {
		return nil, err
	}

	events := parseEvents(messageToString(resp), time.Now())
	if len(events) == 0 {
		return nil, errors.New("anthropic returned no usable event")
	}

	return events, nil
}

// eventMarkers matches, in order of appearance, the markers of the cognition protocol.
var eventMarkers = regexp.MustCompile(
	`(?s)` + regexp.QuoteMeta(thoughtOpenMarker) + `(.*?)` + regexp.QuoteMeta(thoughtCloseMarker) +
		`|` + regexp.QuoteMeta(actionOpenMarker) + `(.*?)` + regexp.QuoteMeta(actionCloseMarker) +
		`|` + regexp.QuoteMeta(doneMarker),
)

// parseEvents splits a model's answer into the events it contains, in order:
// each thought and action marker is an event, and the text before a done marker is
// the final answer that completes the cognition. Text outside markers that does not
// lead to a done marker is kept as a thought, so nothing the model said is lost.
func parseEvents(text string, now time.Time) []Event {
	source := EventSource{Type: EventSourceTypeAgent, ID: anthropic.ModelClaudeHaiku4_5}
	newEvent := func(t EventType, payload string) Event {
		return Event{Type: t, Source: source, Payload: []byte(strings.TrimSpace(payload)), Date: now}
	}

	var events []Event
	last := 0
	for _, m := range eventMarkers.FindAllStringSubmatchIndex(text, -1) {
		residual := strings.TrimSpace(text[last:m[0]])
		last = m[1]

		switch {
		case m[2] >= 0: // thought
			if residual != "" {
				events = append(events, newEvent(EventTypeThought, residual))
			}
			events = append(events, newEvent(EventTypeThought, text[m[2]:m[3]]))
		case m[4] >= 0: // action
			if residual != "" {
				events = append(events, newEvent(EventTypeThought, residual))
			}
			events = append(events, newEvent(EventTypeActionRequest, text[m[4]:m[5]]))
		default: // done
			events = append(events, newEvent(EventTypeCognitionCompleted, residual))
		}
	}

	if tail := strings.TrimSpace(text[last:]); tail != "" {
		events = append(events, newEvent(EventTypeThought, tail))
	}

	return events
}

func mapEventHistoryToEventParams(history []Event) []anthropic.MessageParam {
	params := make([]anthropic.MessageParam, 0, len(history))
	for _, he := range history {
		params = append(params, mapEventToMessageParam(he))
	}

	// The API expects the conversation to end on a user turn.
	if len(params) > 0 && params[len(params)-1].Role == anthropic.MessageParamRoleAssistant {
		params = append(params, anthropic.NewUserMessage(anthropic.NewTextBlock("Continue your reasoning.")))
		//params = append(params, anthropic.NewAssistantMessage(anthropic.NewTextBlock("Note to self, I need to continue my reasoning, and think deeper")))
	}

	return params
}

func mapEventToMessageParam(e Event) anthropic.MessageParam {
	text := FormatEventToProtocol(e)
	if e.Source.Type == EventSourceTypeAgent {
		// The model's own earlier thoughts and proposed actions.
		return anthropic.NewAssistantMessage(anthropic.NewTextBlock(text))
	}
	// Real human input.
	return anthropic.NewUserMessage(anthropic.NewTextBlock(text))
}

// messageToString is the text that the model wrote.
func messageToString(message *anthropic.Message) string {
	var text strings.Builder
	for _, block := range message.Content {
		if t, ok := block.AsAny().(anthropic.TextBlock); ok {
			text.WriteString(t.Text)
		}
	}
	return text.String()
}
