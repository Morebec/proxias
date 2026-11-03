// Command agent-cli is a chat with an agent in the terminal: each line you type is a message to it, and it answers,
// using its actions when it needs to. The agent is set up in code below, on the ReAct strategy over Claude
// (claude-haiku-5-5, its key ANTHROPIC_API_KEY, as the other examples), which thinks adaptively and shows a summary of
// its thinking; or, with -vendor openai, over an OpenAI model (-model, gpt-5-mini by default; its key OPENAI_API_KEY).
//
//	go run ./examples/agent-cli
//	go run ./examples/agent-cli -vendor openai
//	> what time is it?
//	> remember that the demo is on friday
//	> what do you remember?
//
// exit (or Ctrl-D) quits; Ctrl-C cancels the current turn.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	openaisdk "github.com/openai/openai-go/v3"

	"github.com/morebec/proxias/agentsdk"
	agentanthropic "github.com/morebec/proxias/agentsdk/anthropic"
	agentopenai "github.com/morebec/proxias/agentsdk/openai"
	"github.com/morebec/proxias/examples/agentdemo"
)

var (
	vendor = flag.String("vendor", "anthropic", "the model's vendor: anthropic or openai")
	model  = flag.String("model", "", "the model: "+agentdemo.ClaudeModel+" for anthropic, gpt-5-mini for openai, by default")
)

func main() {
	flag.Parse()
	var (
		mutex sync.Mutex
		notes = []string{} // empty, not null, when nothing was remembered yet
	)
	agent := &agentsdk.Agent{
		ID: "cleo",
		AgentPersona: agentsdk.AgentPersona{
			Name:  "Cleo",
			Role:  "personal assistant in a terminal, brief and friendly",
			Goals: []string{"answer the user", "keep the notes the user asks to remember"},
		},
		ActionExecutor: agentsdk.SimpleActionExecutor{Actions: map[string]agentsdk.SimpleAction{
			"get_current_time": {
				ActionDescriptor: agentsdk.ActionDescriptor{ID: "get_current_time", Description: "Tells the current date and time."},
				RunFn: func(context.Context, agentsdk.Session, map[string]any) (agentsdk.ActionExecutionResult, error) {
					return agentsdk.ActionExecutionResult{ActionID: "get_current_time", Status: "completed", Data: map[string]any{"now": time.Now().Format(time.RFC1123)}}, nil
				},
			},
			"remember": {
				ActionDescriptor: agentsdk.ActionDescriptor{ID: "remember", Description: "Keeps a note for later.", InputSchema: agentdemo.Params("note", "what to remember")},
				RunFn: func(_ context.Context, _ agentsdk.Session, params map[string]any) (agentsdk.ActionExecutionResult, error) {
					mutex.Lock()
					defer mutex.Unlock()
					notes = append(notes, fmt.Sprint(params["note"]))
					return agentsdk.ActionExecutionResult{ActionID: "remember", Status: "completed"}, nil
				},
			},
			"recall": {
				ActionDescriptor: agentsdk.ActionDescriptor{ID: "recall", Description: "Lists the notes kept so far."},
				RunFn: func(context.Context, agentsdk.Session, map[string]any) (agentsdk.ActionExecutionResult, error) {
					mutex.Lock()
					defer mutex.Unlock()
					return agentsdk.ActionExecutionResult{ActionID: "recall", Status: "completed", Data: map[string]any{"notes": notes}}, nil
				},
			},
		}},
	}
	var llm agentsdk.LLM
	switch *vendor {
	case "anthropic":
		if *model == "" {
			*model = agentdemo.ClaudeModel
		}
		// Claude thinks adaptively, and says what it thought: a summary of its thinking.
		claude := agentanthropic.NewLLM(agentdemo.Client(), *model)
		claude.Thinking = anthropic.ThinkingConfigParamUnion{OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{Display: anthropic.ThinkingConfigAdaptiveDisplaySummarized}}
		llm = claude
	case "openai":
		if *model == "" {
			*model = "gpt-5-mini"
		}
		llm = agentopenai.NewLLM(openaisdk.NewClient(), *model) // its key: OPENAI_API_KEY
	default:
		fmt.Fprintf(os.Stderr, "-vendor %q: it is anthropic or openai\n", *vendor)
		os.Exit(2)
	}
	strategy := agentsdk.NewReActStrategy(llm)

	fmt.Println("Talking to Cleo (" + *vendor + ", " + *model + "). exit quits, Ctrl-C cancels a turn.")
	session := agentsdk.NewSession()
	lines := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !lines.Scan() {
			fmt.Println()
			return
		}
		text := strings.TrimSpace(lines.Text())
		if text == "" {
			continue
		}
		if text == "exit" {
			return
		}

		// The turn, cancelled by Ctrl-C.
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
		stream, err := agent.Loop(ctx, session.WithEvents(agentsdk.NewMessageEvent(agentsdk.NewUserSource("you"), text)), strategy)
		if err != nil {
			cancel()
			fmt.Fprintln(os.Stderr, err)
			return
		}
		// The events as they come (not the chunks: a text is a thought or the reply only once its event says so).
		for update := range stream.Updates() {
			if update.Err != nil {
				fmt.Fprintln(os.Stderr, "error:", update.Err)
				continue
			}
			for _, e := range update.Events {
				switch {
				case e.Type == agentsdk.EventTypeThought:
					// Claude's thinking is kept whole, as JSON: its text is what to show.
					thought := e.Text()
					var thinking agentsdk.LLMChunk
					if json.Unmarshal(e.Payload, &thinking) == nil && thinking.Signature != "" {
						thought = thinking.Thinking
					}
					if thought = strings.TrimSpace(thought); thought != "" {
						fmt.Printf("\033[2;3mthinking: %s\033[0m\n", thought)
					}
				case e.Type == agentsdk.EventTypeActionRequest || e.Type == agentsdk.EventTypeActionResult:
					fmt.Printf("\033[2m%s %s\033[0m\n", e.Type, e.Text())
				case e.Source == agent.AsSource():
					if text := strings.TrimSpace(e.Text()); text != "" {
						fmt.Println(text)
					}
				}
			}
		}
		cancel()
		session = stream.Session()
	}
}
