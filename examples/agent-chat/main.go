// Command agent-chat is an assistant with one action, chatted with in one session, or served over HTTP (-serve) with
// one of the transports (-transport). It runs on a script; -claude runs it on Claude, -strategy=gated on the gated
// strategy.
//
//	go run ./examples/agent-chat
//
//	go run ./examples/agent-chat -serve :8080
//	curl -N localhost:8080/sessions/s1/messages -d '{"from":"john","text":"Email joe@boss.com: Hello Boss!"}'
//
//	go run ./examples/agent-chat -serve :8080 -transport openai
//	curl localhost:8080/v1/chat/completions -d '{"model":"ada","messages":[{"role":"user","content":"Email joe@boss.com: Hello Boss!"}]}'
//
//	go run ./examples/agent-chat -serve :8080 -transport agentprotocol
//	curl localhost:8080/ap/v1/agent/tasks -d '{"input":"Email joe@boss.com: Hello Boss!"}'
//	curl -X POST localhost:8080/ap/v1/agent/tasks/<task_id>/steps
package main

import (
	"context"
	"flag"
	"fmt"
	"iter"
	"log"
	"net/http"
	"time"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/agentsdk"
	"github.com/morebec/proxias/agentsdk/transports/agenthttp"
	"github.com/morebec/proxias/agentsdk/transports/agentprotocolhttp"
	"github.com/morebec/proxias/agentsdk/transports/openaihttp"
	"github.com/morebec/proxias/examples/agentdemo"
)

var (
	serve     = flag.String("serve", "", "serve the agent over HTTP on this address instead of chatting")
	transport = flag.String("transport", "agenthttp", "the transport of -serve: agenthttp, openai (the Chat Completions API) or agentprotocol (agentprotocol.ai)")
)

func main() {
	flag.Parse()

	ada := agentsdk.Agent{
		ID:           "ada",
		AgentPersona: agentsdk.AgentPersona{Name: "Ada", Role: "personal assistant"},
		ActionExecutor: agentdemo.Actions(agentdemo.Action{
			ActionDescriptor: agentsdk.ActionDescriptor{ID: "send_email", Description: "Sends an email.", InputSchema: agentdemo.Params("to", "the address", "body", "the text")},
			Run: func(_ context.Context, params map[string]any) (any, error) {
				return fmt.Sprintf("sent to %v", params["to"]), nil
			},
		}),
		Middleware: []agentsdk.AgentMiddleware{withClock}, // an extension: every turn is told the time
	}
	strategy := agentdemo.Strategy(agentdemo.Script{
		React: []string{
			`<thought>An email to send.</thought><action>{"action_id": "send_email", "params": {"to": "joe@boss.com", "body": "Hello Boss!"}}</action>`,
			`Done: I emailed "Hello Boss!" to joe@boss.com.` + agentdemo.Done,
			`You're welcome!` + agentdemo.Done,
		},
		Intents:   []proxias.Intent{agentdemo.Intent("send_email", "to", "joe@boss.com", "body", "Hello Boss!")},
		Syntheses: []string{`Done: I emailed "Hello Boss!" to joe@boss.com.`},
		Fallback:  []string{`You're welcome!` + agentdemo.Done},
	})

	if *serve != "" {
		handlers := map[string]http.Handler{
			"agenthttp":     agenthttp.NewHandler(ada, strategy),
			"openai":        openaihttp.NewHandler(strategy, []agentsdk.Agent{ada}),
			"agentprotocol": agentprotocolhttp.NewHandler(ada, strategy),
		}
		handler, ok := handlers[*transport]
		if !ok {
			log.Fatalf("-transport %q: it is agenthttp, openai or agentprotocol", *transport)
		}
		log.Fatal(http.ListenAndServe(*serve, handler))
	}

	session := agentsdk.NewSession()
	for _, text := range []string{"Please email joe@boss.com: Hello Boss!", "Thanks!"} {
		message := agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), text)
		agentdemo.Print(message)
		stream, err := ada.Loop(context.Background(), session.WithEvents(message), strategy)
		if err != nil {
			log.Fatal(err)
		}
		for update := range stream.Updates() {
			if update.Err != nil {
				log.Fatal(update.Err)
			}
			for _, e := range update.Events {
				agentdemo.Print(e)
			}
		}
		session = stream.Session()
	}
}

// withClock tells the turn the time it is, as a system message of its session for this turn only.
func withClock(next agentsdk.CognitiveStrategy) agentsdk.CognitiveStrategy {
	return agentsdk.StrategyFunc(func(ctx context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
		inv.Session = inv.Session.WithEvents(agentsdk.NewMessageEvent(agentsdk.NewSystemSource("clock"), "It is "+time.Now().Format(time.RFC1123)+"."))
		return next.Invoke(ctx, inv)
	})
}
