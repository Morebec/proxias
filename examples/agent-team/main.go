// Command agent-team has agents work together, orchestrated by the app: a manager delegates to a researcher (asking it
// is one of the manager's actions, agentdemo.Ask), then two agents talk in turn. It runs on scripts; -claude runs it
// on Claude, -strategy=gated on the gated strategy.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/agentsdk"
	"github.com/morebec/proxias/examples/agentdemo"
)

func main() {
	ctx := context.Background()

	// Delegation: asking rita is one of max's actions.
	ritaSays := `Go 1.25 added sync.WaitGroup.Go, which starts a goroutine and tracks it.` + agentdemo.Done
	rita := agentsdk.Agent{ID: "rita", AgentPersona: agentsdk.AgentPersona{Name: "Rita", Role: "researcher, answers in one sentence"}}
	max := agentsdk.Agent{ID: "max", AgentPersona: agentsdk.AgentPersona{Name: "Max", Role: "team lead, delegates research"}}
	max.ActionExecutor = agentdemo.Ask(max, rita, agentdemo.Strategy(agentdemo.Script{React: []string{ritaSays}, Reasoning: []string{ritaSays}}))
	question := agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "What's new in sync.WaitGroup in Go 1.25? Ask Rita.")
	agentdemo.Print(question)
	asked := agentsdk.NewSession(question)
	stream, err := max.Loop(ctx, asked, agentdemo.Strategy(agentdemo.Script{
		React: []string{
			`<action>{"action_id": "ask_rita", "params": {"message": "What did Go 1.25 add to sync.WaitGroup?"}}</action>`,
			`Rita says Go 1.25 added WaitGroup.Go.` + agentdemo.Done,
		},
		Intents:   []proxias.Intent{agentdemo.Intent("ask_rita", "message", "What did Go 1.25 add to sync.WaitGroup?")},
		Syntheses: []string{`Rita says Go 1.25 added WaitGroup.Go.`},
	}))
	if err != nil {
		log.Fatal(err)
	}
	ended, err := stream.Collect()
	for _, e := range ended.Events()[len(asked.Events()):] {
		agentdemo.Print(e)
	}
	if err != nil {
		log.Fatal(err)
	}

	// Conversation: two agents reply to each other, four times in all.
	fmt.Println()
	olga := agentsdk.Agent{ID: "olga", AgentPersona: agentsdk.AgentPersona{Name: "Olga", Role: "optimist, one sentence per reply"}}
	sam := agentsdk.Agent{ID: "sam", AgentPersona: agentsdk.AgentPersona{Name: "Sam", Role: "skeptic, one sentence per reply"}}
	says := []string{
		`Rewriting it in Go will make it faster.` + agentdemo.Done,
		`Only if the bottleneck is the CPU, not the database.` + agentdemo.Done,
		`Then we measure first, and rewrite the hot path.` + agentdemo.Done,
		`Agreed.` + agentdemo.Done,
	}
	opener := agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "Should we rewrite the billing service in Go?")
	agentdemo.Print(opener)
	if err := converse(ctx, opener, 4, []agentsdk.Agent{olga, sam}, agentdemo.Strategy(agentdemo.Script{React: says, Reasoning: says})); err != nil {
		log.Fatal(err)
	}
}

// converse has the agents talk in turn, rounds times in all: the first is told the opener, each after it what the one
// before it replied, as a message of that agent, each in its own session. It ends early when an agent says nothing.
func converse(ctx context.Context, opener agentsdk.Event, rounds int, agents []agentsdk.Agent, strategy agentsdk.CognitiveStrategy) error {
	sessions := make([]agentsdk.Session, len(agents))
	for i := range sessions {
		sessions[i] = agentsdk.NewSession()
	}
	said := opener
	for round := range rounds {
		i := round % len(agents)
		told := sessions[i].WithEvents(said)
		stream, err := agents[i].Loop(ctx, told, strategy)
		if err != nil {
			return err
		}
		ended, err := stream.Collect()
		sessions[i] = ended
		if err != nil {
			return err
		}
		reply := agentdemo.Reply(ended.Events()[len(told.Events()):])
		if reply == "" {
			return nil
		}
		said = agentsdk.NewMessageEvent(agents[i].AsSource(), reply)
		agentdemo.Print(said)
	}
	return nil
}
