package agentdemo

import (
	"regexp"
	"strings"

	"github.com/morebec/proxias"
)

// The markers of a scripted answer: <thought>...</thought>,
// <action>{"action_id": ..., "params": ...}</action>, and a
// final answer ended by Done.
const (
	Done = "<done/>"

	actionOpen   = "<action>"
	actionClose  = "</action>"
	thoughtOpen  = "<thought>"
	thoughtClose = "</thought>"
)

// markers matches the markers of a scripted answer, in order of appearance.
var markers = regexp.MustCompile(
	`(?s)` + regexp.QuoteMeta(
		thoughtOpen,
	) + `(.*?)` + regexp.QuoteMeta(
		thoughtClose,
	) +
		`|` + regexp.QuoteMeta(
		actionOpen,
	) + `(.*?)` + regexp.QuoteMeta(
		actionClose,
	) +
		`|` + regexp.QuoteMeta(
		Done,
	),
)

// parseAnswer reads the events of a scripted answer, in order: a thought, an
// action request, and the text before Done as
// the reply.
func parseAnswer(answer string) []proxias.Event {
	event := func(t proxias.EventType, payload string) proxias.Event {
		return proxias.Event{
			Type:    t,
			Payload: []byte(strings.TrimSpace(payload)),
		}
	}
	var events []proxias.Event
	last := 0
	for _, m := range markers.FindAllStringSubmatchIndex(answer, -1) {
		residual := strings.TrimSpace(answer[last:m[0]])
		last = m[1]
		switch {
		case m[2] >= 0:
			events = append(events, event(proxias.EventTypeThought, answer[m[2]:m[3]]))
		case m[4] >= 0:
			if residual != "" {
				events = append(events, event(proxias.EventTypeThought, residual))
			}
			events = append(events, event(proxias.EventTypeActionRequest, answer[m[4]:m[5]]))
		default:
			events = append(events, event(proxias.EventTypeMessage, residual))
		}
	}
	return events
}
