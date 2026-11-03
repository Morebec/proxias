// Command agent-unattended runs an agent with nobody talking to it: it watches
// a probe of a system, and decides on its own whether to act. It runs on a
// script; -claude runs it on Claude, -strategy=gated on the gated strategy.
package main

import (
	"context"
	"log"
	"time"

	"github.com/morebec/proxias/agentsdk"
	"github.com/morebec/proxias/examples/agentdemo"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	otto := agentsdk.Agent{
		ID: "otto",
		AgentPersona: agentsdk.AgentPersona{
			Name:  "Otto",
			Role:  "watches the servers",
			Goals: []string{"page on call only for what needs a person"},
		},
		ActionExecutor: agentdemo.Actions(agentdemo.Action{
			ActionDescriptor: agentsdk.ActionDescriptor{
				ID:          "page_oncall",
				Description: "Pages the engineer on call.",
				InputSchema: agentdemo.Params("reason", "why, in one sentence"),
			},
			Run: func(context.Context, map[string]any) (any, error) {
				return "paged", nil
			},
		}),
	}
	strategy := agentdemo.Strategy(agentdemo.Script{
		React: []string{
			`<thought>Nothing to do.</thought>` + agentdemo.Done,
			`<action>{"action_id": "page_oncall", "params": {"reason": "the ` +
				`disk of db-1 is 93% full"}}</action>`,
			`I paged on call about db-1's disk.` + agentdemo.Done,
		},
		Requests: [][]agentsdk.ActionExecutionRequest{
			{},
			agentdemo.Requests("page_oncall", "reason", "the disk of db-1 is 93% full"),
		},
		Syntheses: []string{`I paged on call about db-1's disk.`},
		Reasoning: []string{
			`<thought>Nothing to do.</thought>` + agentdemo.Done,
		},
	})

	// What it watches: a probe of db-1's disk, every second.
	probe := make(chan agentsdk.Event)
	go func() {
		defer close(probe)
		for _, reading := range []string{"disk 41% full", "disk 93% full"} {
			select {
			case <-time.After(time.Second):
				db := agentsdk.NewSystemSource("db-1")
				probe <- agentsdk.NewEvent("probe.disk", db, []byte(reading))
			case <-ctx.Done():
				return
			}
		}
	}()

	stream, err := otto.Watch(ctx, agentsdk.NewSession(), probe, strategy)
	if err != nil {
		log.Print(err)
		return
	}
	for d := range stream.Updates() {
		for _, e := range d.Events {
			agentdemo.Print(e)
		}
		if d.Err != nil && ctx.Err() == nil {
			agentdemo.Print(agentsdk.NewMessageEvent(agentsdk.NewSystemSource("watch"), d.Err.Error()))
		}
	}
}
