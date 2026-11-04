// Command agent-team has agents work together, orchestrated by the app: a
// manager delegates to a researcher (asking it is one of the manager's actions,
// agentdemo.Ask), then two agents talk in turn. It runs on scripts; -claude
// runs it
// on Claude, -strategy on another strategy (-strategy=adaptive, say).
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/examples/agentdemo"
)

func main() {
	ctx := context.Background()

	// Delegation: asking rita is one of max's actions.
	ritaSays := `Go 1.25 added sync.WaitGroup.Go, which starts a goroutine ` +
		`and tracks it.` + agentdemo.Done
	rita := proxias.Agent{
		ID: "rita",
		AgentPersona: proxias.AgentPersona{
			Name: "Rita",
			Role: "researcher, answers in one sentence",
		},
	}
	max := proxias.Agent{
		ID: "max",
		AgentPersona: proxias.AgentPersona{
			Name: "Max",
			Role: "team lead, delegates research",
		},
	}
	max.ActionExecutor = agentdemo.Ask(
		max,
		rita,
		agentdemo.Strategy(
			agentdemo.Script{
				React:     []string{ritaSays},
				Reasoning: []string{ritaSays},
			},
		),
	)
	question := proxias.NewMessageEvent(
		proxias.NewUserSource("john"),
		"What's new in sync.WaitGroup in Go 1.25? Ask Rita.",
	)
	agentdemo.Print(question)
	asked := proxias.NewSession(question)
	stream, err := max.Loop(ctx, asked, agentdemo.Strategy(agentdemo.Script{
		React: []string{
			`<action>{"action_id": "ask_rita", "params": {"message": "What ` +
				`did Go 1.25 add to sync.WaitGroup?"}}</action>`,
			`Rita says Go 1.25 added WaitGroup.Go.` + agentdemo.Done,
		},
		Requests: [][]proxias.ActionExecutionRequest{
			agentdemo.Requests("ask_rita", "message", "What did Go 1.25 add to sync.WaitGroup?"),
		},
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
	olga := proxias.Agent{
		ID: "olga",
		AgentPersona: proxias.AgentPersona{
			Name: "Olga",
			Role: "optimist, one sentence per reply",
		},
	}
	sam := proxias.Agent{
		ID: "sam",
		AgentPersona: proxias.AgentPersona{
			Name: "Sam",
			Role: "skeptic, one sentence per reply",
		},
	}
	says := []string{
		`Rewriting it in Go will make it faster.` + agentdemo.Done,
		`Only if the bottleneck is the CPU, not the database.` + agentdemo.Done,
		`Then we measure first, and rewrite the hot path.` + agentdemo.Done,
		`Agreed.` + agentdemo.Done,
	}
	opener := proxias.NewMessageEvent(proxias.NewUserSource("john"), "Should we rewrite the billing service in Go?")
	agentdemo.Print(opener)
	strategy := agentdemo.Strategy(agentdemo.Script{
		React:     says,
		Reasoning: says,
	})
	agents := []proxias.Agent{olga, sam}
	if err := converse(ctx, opener, 4, agents, strategy); err != nil {
		log.Fatal(err)
	}
}

// converse has the agents talk in turn, rounds times in all: the first is told
// the opener, each after it what the one before it replied, as a message of
// that agent, each in its own session. It ends early when an agent says
// nothing.
func converse(
	ctx context.Context,
	opener proxias.Event,
	rounds int,
	agents []proxias.Agent,
	strategy proxias.CognitiveStrategy,
) error {
	sessions := make([]proxias.Session, len(agents))
	for i := range sessions {
		sessions[i] = proxias.NewSession()
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
		said = proxias.NewMessageEvent(agents[i].AsSource(), reply)
		agentdemo.Print(said)
	}
	return nil
}
