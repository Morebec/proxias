package agentdemo

import (
	"context"
	"sync"

	"github.com/morebec/proxias/agentsdk"
)

// Ask is how the examples have an agent talk to another: an action of the app,
// "ask_<to.ID>", whose parameter "message" is told to the agent to, as a
// message of the agent from, Looped with the strategy; its reply is the output
// (Data "reply"). Each session of from keeps its own session with to, so to
// remembers what it was asked.
func Ask(from, to agentsdk.Agent, strategy agentsdk.CognitiveStrategy) agentsdk.ActionExecutor {
	var (
		mutex    sync.Mutex
		sessions = map[string]agentsdk.Session{}
	)
	id := "ask_" + to.ID
	return agentsdk.SimpleActionExecutor{
		Actions: map[string]agentsdk.SimpleAction{id: {
			ActionDescriptor: agentsdk.ActionDescriptor{
				ID:   id,
				Name: id,
				Description: "Asks " + to.Name + " (" + to.Role + ") " +
					"something, and gives its reply.",
				InputSchema: Params(
					"message",
					"what to ask or tell it, in full: it knows nothing else "+
						"of your work",
				),
			},
			RunFn: func(
				ctx context.Context,
				s agentsdk.Session,
				params map[string]any,
			) (agentsdk.ActionExecutionResult, error) {
				message, _ := params["message"].(string)
				if message == "" {
					return agentsdk.ActionExecutionResult{
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
					asked = agentsdk.NewSession()
				}
				asked = asked.WithEvents(agentsdk.NewMessageEvent(from.AsSource(), message))
				stream, err := to.Loop(ctx, asked, strategy)
				if err != nil {
					return agentsdk.ActionExecutionResult{}, err
				}
				ended, err := stream.Collect()
				sessions[s.ID()] = ended
				if err != nil {
					return agentsdk.ActionExecutionResult{}, err
				}
				return agentsdk.ActionExecutionResult{
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
func Reply(events []agentsdk.Event) string {
	for i := len(events) - 1; i >= 0; i-- {
		e := events[i]
		if !e.HasSourceType(agentsdk.EventSourceTypeAgent) || e.Type != agentsdk.EventTypeMessage {
			continue
		}
		if len(e.Payload) > 0 {
			return e.Text()
		}
	}
	return ""
}
