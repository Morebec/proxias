package agentdemo

import (
	"context"
	"sync"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/schema"
)

// Ask is how the examples have an agent talk to another: an action of the app,
// "ask_<to.ID>", whose parameter "message" is told to the agent to, as a
// message of the agent from, Looped with the strategy; its reply is the output
// (Data "reply"). Each session of from keeps its own session with to, so to
// remembers what it was asked.
func Ask(from, to proxias.Agent, strategy proxias.CognitiveStrategy) proxias.ActionExecutor {
	var (
		mutex    sync.Mutex
		sessions = map[string]proxias.Session{}
	)
	id := "ask_" + to.ID
	return proxias.SimpleActionExecutor{
		Actions: map[string]proxias.SimpleAction{id: {
			ActionDescriptor: proxias.ActionDescriptor{
				ID:   id,
				Name: id,
				Description: "Asks " + to.Name + " (" + to.Role + ") " +
					"something, and gives its reply.",
				InputSchema: schema.Object().
					String("message", "what to ask or tell it, in full: it knows nothing else "+
						"of your work").Required().
					Map(),
			},
			RunFn: func(
				ctx context.Context,
				s proxias.Session,
				params map[string]any,
			) (proxias.ActionExecutionResult, error) {
				message, _ := params["message"].(string)
				if message == "" {
					return proxias.ActionExecutionResult{
						ActionID: id,
						Status:   "failed",
						Data: map[string]any{
							"error": `the parameter "message" is required`,
						},
					}, nil
				}
				mutex.Lock()
				defer mutex.Unlock()
				asked, ok := sessions[s.ID()]
				if !ok {
					asked = proxias.NewSession()
				}
				asked = asked.WithEvents(proxias.NewMessageEvent(from.AsSource(), message))
				stream, err := to.Loop(ctx, asked, strategy)
				if err != nil {
					return proxias.ActionExecutionResult{}, err
				}
				ended, err := stream.Collect()
				sessions[s.ID()] = ended
				if err != nil {
					return proxias.ActionExecutionResult{}, err
				}
				return proxias.ActionExecutionResult{
					ActionID: id,
					Status:   "completed",
					Data: map[string]any{
						"reply": Reply(ended.Events()[len(asked.Events()):]),
					},
				}, nil
			},
		}},
	}
}

// Reply is what the events say to whoever reads them: the text of the last
// message of an agent that
// says something, "" for none.
func Reply(events []proxias.Event) string {
	for i := len(events) - 1; i >= 0; i-- {
		e := events[i]
		if !e.HasSourceType(proxias.EventSourceTypeAgent) || e.Type != proxias.EventTypeMessage {
			continue
		}
		if len(e.Payload) > 0 {
			return e.Text()
		}
	}
	return ""
}
