package main

import (
	"context"
	"fmt"
	"slices"
	"sync"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/agentsdk"
	"github.com/morebec/proxias/examples/agentdemo"
)

// all are the scenarios of the examples, each agent with its own action.
func all() []scenario {
	return []scenario{
		{name: "chat", run: chat},
		{name: "delegate", run: delegate},
		{name: "unattended", run: unattended},
	}
}

// chat is agent-chat's: the email must be sent to joe@boss.com, once, and the thanks answered.
func chat(ctx context.Context, r *run) (bool, string, error) {
	var (
		mutex sync.Mutex
		sent  []any
	)
	ada := agentsdk.Agent{
		ID:           "ada",
		AgentPersona: agentsdk.AgentPersona{Name: "Ada", Role: "personal assistant"},
		ActionExecutor: agentdemo.Actions(agentdemo.Action{
			ActionDescriptor: agentsdk.ActionDescriptor{ID: "send_email", Description: "Sends an email.", InputSchema: agentdemo.Params("to", "the address", "body", "the text")},
			Run: func(_ context.Context, params map[string]any) (any, error) {
				mutex.Lock()
				defer mutex.Unlock()
				sent = append(sent, params["to"])
				return fmt.Sprintf("sent to %v", params["to"]), nil
			},
		}),
	}
	strategy := r.strategy(agentdemo.Script{
		React: []string{
			`<action>{"action_id": "send_email", "params": {"to": "joe@boss.com", "body": "Hello Boss!"}}</action>`,
			`I emailed joe@boss.com.` + agentdemo.Done,
			`You're welcome!` + agentdemo.Done,
		},
		Intents:   []proxias.Intent{agentdemo.Intent("send_email", "to", "joe@boss.com", "body", "Hello Boss!")},
		Syntheses: []string{`I emailed joe@boss.com.`},
		Fallback:  []string{`You're welcome!` + agentdemo.Done},
	})
	thanked, err := r.chat(ctx, ada, strategy, "Please email joe@boss.com: Hello Boss!", "Thanks!")
	if err != nil {
		return false, "", err
	}
	switch {
	case len(sent) != 1 || sent[0] != "joe@boss.com":
		return false, fmt.Sprintf("emails sent to %v", sent), nil
	case thanked == "":
		return false, "the thanks were not answered", nil
	}
	return true, "", nil
}

// delegate is agent-team's delegation: rita must be asked, reply, and max answer.
func delegate(ctx context.Context, r *run) (bool, string, error) {
	ritaSays := `Go 1.25 added sync.WaitGroup.Go, which starts a goroutine and tracks it.` + agentdemo.Done
	rita := agentsdk.Agent{ID: "rita", AgentPersona: agentsdk.AgentPersona{Name: "Rita", Role: "researcher, answers in one sentence"}}
	max := agentsdk.Agent{ID: "max", AgentPersona: agentsdk.AgentPersona{Name: "Max", Role: "team lead, delegates research"}}
	max.ActionExecutor = agentdemo.Ask(max, rita, r.strategy(agentdemo.Script{React: []string{ritaSays}, Fallback: []string{ritaSays}}))
	strategy := r.strategy(agentdemo.Script{
		React: []string{
			`<action>{"action_id": "ask_rita", "params": {"message": "What did Go 1.25 add to sync.WaitGroup?"}}</action>`,
			`Rita says Go 1.25 added WaitGroup.Go.` + agentdemo.Done,
		},
		Intents:   []proxias.Intent{agentdemo.Intent("ask_rita", "message", "What did Go 1.25 add to sync.WaitGroup?")},
		Syntheses: []string{`Rita says Go 1.25 added WaitGroup.Go.`},
	})
	question := agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "What's new in sync.WaitGroup in Go 1.25? Ask Rita.")
	_, events, err := r.loop(ctx, max, agentsdk.NewSession(question), strategy)
	if err != nil {
		return false, "", err
	}
	replied := false
	for _, result := range results(events, "ask_rita") {
		if reply, _ := result.Data["reply"].(string); reply != "" {
			replied = true
		}
	}
	switch {
	case !replied:
		return false, "rita was not asked, or did not reply", nil
	case agentdemo.Reply(events) == "":
		return false, "max did not answer", nil
	}
	return true, "", nil
}

// unattended is agent-unattended's, its events given in order: three readings of a disk, the second full. On call
// must be paged once, for it.
func unattended(ctx context.Context, r *run) (bool, string, error) {
	otto := agentsdk.Agent{
		ID:           "otto",
		AgentPersona: agentsdk.AgentPersona{Name: "Otto", Role: "watches the servers", Goals: []string{"page on call when a disk is over 90% full"}},
		ActionExecutor: agentdemo.Actions(agentdemo.Action{
			ActionDescriptor: agentsdk.ActionDescriptor{ID: "page_oncall", Description: "Pages the engineer on call.", InputSchema: agentdemo.Params("reason", "why, in one sentence")},
			Run:              func(context.Context, map[string]any) (any, error) { return "paged", nil },
		}),
	}
	strategy := r.strategy(agentdemo.Script{
		React: []string{
			`<thought>Nothing to do.</thought>` + agentdemo.Done,
			`<action>{"action_id": "page_oncall", "params": {"reason": "the disk of db-1 is 93% full"}}</action>`,
			`I paged on call about db-1's disk.` + agentdemo.Done,
			agentdemo.Done,
		},
		Intents:   []proxias.Intent{{}, agentdemo.Intent("page_oncall", "reason", "the disk of db-1 is 93% full")},
		Syntheses: []string{`I paged on call about db-1's disk.`},
		Fallback:  []string{`<thought>Nothing to do.</thought>` + agentdemo.Done},
	})
	given := unattendedEvents()
	session, err := r.watch(ctx, otto, strategy, given...)
	if err != nil {
		return false, "", err
	}
	if paged := resultsAt(session, given, "page_oncall"); !slices.Equal(paged, []int{1}) {
		return false, fmt.Sprintf("paged at events %v", paged), nil
	}
	return true, "", nil
}

// unattendedEvents are what an unattended agent is given, in order: three readings of a disk, the second full.
func unattendedEvents() []agentsdk.Event {
	return []agentsdk.Event{
		agentsdk.NewEvent("probe.disk", agentsdk.NewSystemSource("db-1"), []byte("disk 41% full")),
		agentsdk.NewEvent("probe.disk", agentsdk.NewSystemSource("db-1"), []byte("disk 93% full")),
		agentsdk.NewEvent("probe.disk", agentsdk.NewSystemSource("db-1"), []byte("disk 42% full")),
	}
}

// catalogChat is the chat scenario on the catalog: ada must send the email once, to joe@boss.com, then answer the
// thanks without acting.
func catalogChat(ctx context.Context, r *run) (bool, string, error) {
	c := &calls{}
	ada := withCatalog("ada", agentsdk.AgentPersona{Name: "Ada", Role: "personal assistant"}, c)
	strategy := r.strategy(agentdemo.Script{
		React: []string{
			`<action>{"action_id": "email_send", "params": {"to": "joe@boss.com", "subject": "Hello", "body": "Hello Boss!"}}</action>`,
			"I emailed joe@boss.com." + agentdemo.Done,
			"You're welcome!" + agentdemo.Done,
		},
		Intents:   []proxias.Intent{agentdemo.Intent("email_send", "to", "joe@boss.com", "subject", "Hello", "body", "Hello Boss!")},
		Syntheses: []string{"I emailed joe@boss.com."},
		Fallback:  []string{"You're welcome!" + agentdemo.Done},
	})
	thanked, err := r.chat(ctx, ada, strategy, "Please email joe@boss.com: Hello Boss!", "Thanks!")
	if err != nil {
		return false, "", err
	}
	if ok, why := check(c.all(), expected{"email_send", func(p map[string]any) bool {
		return has(p, "to", "joe@boss.com") && has(p, "body", "hello boss")
	}}); !ok {
		return false, why, nil
	}
	if thanked == "" {
		return false, "the thanks were not answered", nil
	}
	return true, "", nil
}

// catalogUnattended is the unattended scenario on the catalog, its events in order (three readings of a disk, the
// second full): otto must page on call once, for the disk, and do nothing else.
func catalogUnattended(ctx context.Context, r *run) (bool, string, error) {
	c := &calls{}
	idle := "<thought>Nothing to do.</thought>" + agentdemo.Done
	otto := withCatalog("otto", agentsdk.AgentPersona{
		Name:  "Otto",
		Role:  "watches the servers",
		Goals: []string{"page on call when a disk is over 90% full"},
	}, c)
	strategy := r.strategy(agentdemo.Script{
		React: []string{
			idle,
			`<action>{"action_id": "devops_page_oncall", "params": {"reason": "the disk of db-1 is 93% full"}}</action>`,
			"I paged on call about db-1's disk." + agentdemo.Done,
			agentdemo.Done,
		},
		Intents:   []proxias.Intent{{}, agentdemo.Intent("devops_page_oncall", "reason", "the disk of db-1 is 93% full")},
		Syntheses: []string{"I paged on call about db-1's disk."},
		Fallback:  []string{idle, agentdemo.Done},
	})
	if _, err := r.watch(ctx, otto, strategy, unattendedEvents()...); err != nil {
		return false, "", err
	}
	ok, why := check(c.all(), expected{"devops_page_oncall", func(p map[string]any) bool { return has(p, "reason", "disk") }})
	return ok, why, nil
}
