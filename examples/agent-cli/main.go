// Command agent-cli is a chat with an agent in the terminal: each line you type
// is a message to it, and it answers, using its actions when it needs to. The
// agent is set up in code below, on the ReAct strategy over Claude
// (claude-haiku-5-5, its key ANTHROPIC_API_KEY, as the other examples), which
// thinks adaptively and shows a summary of its thinking; or, with -vendor
// openai, over an OpenAI model (-model, gpt-5-mini by default; its key
// OPENAI_API_KEY).
//
// It can also read and write files, and run bash commands, in its workspace
// (-workdir, ./workspace by default): enough to write itself scripts, and so
// new capabilities. Nothing sandboxes them beyond the working directory: run it
// where
// that is fine.
//
//	go run ./examples/agent-cli
//	go run ./examples/agent-cli -vendor openai
//
// It curates its own memory, across runs: MEMORY.md in its workspace, which it
// is told at every turn and rewrites
// with its update_memory action.
//
//	> what time is it?
//	> the demo is on friday
//	(later, in another run)
//	> when is the demo?
//	> write yourself a script that tells the weather in Montreal, then use it
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
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	openaisdk "github.com/openai/openai-go/v3"

	"github.com/morebec/proxias"
	agentanthropic "github.com/morebec/proxias/adapters/anthropic"
	agentopenai "github.com/morebec/proxias/adapters/openai"
	"github.com/morebec/proxias/examples/agentdemo"
)

var (
	vendor = flag.String("vendor", "anthropic", "the model's vendor: anthropic or openai")
	model  = flag.String(
		"model",
		"",
		"the model: "+agentdemo.ClaudeModel+" for anthropic, "+
			"gpt-5-mini for openai, by default",
	)
	workdir = flag.String(
		"workdir",
		"workspace",
		"the directory where the agent reads, writes and runs bash; created "+
			"if missing",
	)
)

func main() {
	flag.Parse()
	dir, err := filepath.Abs(*workdir)
	if err == nil {
		err = os.MkdirAll(dir, 0o750)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "-workdir:", err)
		os.Exit(2)
	}
	// path is where a path of the agent is: in the workspace, unless absolute.
	path := func(params map[string]any) string {
		p := fmt.Sprint(params["path"])
		if filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(dir, p)
	}
	agent := &proxias.Agent{
		ID: "cleo",
		AgentPersona: proxias.AgentPersona{
			Name: "Cleo",
			Role: "personal assistant in a terminal, brief and friendly",
			Goals: []string{
				"answer the user",
				"give yourself the capabilities you lack",
			},
		},
		Memory: proxias.MarkdownMemory{Path: filepath.Join(dir, "MEMORY.md")},
		ActionExecutor: proxias.SimpleActionExecutor{
			Actions: map[string]proxias.SimpleAction{
				"get_current_time": {
					ActionDescriptor: proxias.ActionDescriptor{
						ID:          "get_current_time",
						Description: "Tells the current date and time.",
					},
					RunFn: func(
						context.Context,
						proxias.Session,
						map[string]any,
					) (proxias.ActionExecutionResult, error) {
						return proxias.ActionExecutionResult{
							ActionID: "get_current_time",
							Status:   "completed",
							Data: map[string]any{
								"now": time.Now().Format(time.RFC1123),
							},
						}, nil
					},
				},
				"read_file": {
					ActionDescriptor: proxias.ActionDescriptor{
						ID: "read_file",
						Description: "Reads a file of the workspace (a path " +
							"relative to it, or absolute).",
						InputSchema: agentdemo.Params("path", "the path of the file"),
					},
					RunFn: func(
						_ context.Context,
						_ proxias.Session,
						params map[string]any,
					) (proxias.ActionExecutionResult, error) {
						content, err := os.ReadFile(path(params))
						if err != nil {
							//nolint:nilerr // its result says why
							return proxias.ActionExecutionResult{
								ActionID: "read_file",
								Status:   "failed",
								Data:     map[string]any{"error": err.Error()},
							}, nil
						}
						return proxias.ActionExecutionResult{
							ActionID: "read_file",
							Status:   "completed",
							Data: map[string]any{
								"content": string(content),
							},
						}, nil
					},
				},
				"write_file": {
					ActionDescriptor: proxias.ActionDescriptor{
						ID: "write_file",
						Description: "Writes a file of the workspace, " +
							"creating it and its directories, or replacing " +
							"it.",
						InputSchema: agentdemo.Params(
							"path",
							"the path of the file",
							"content",
							"the whole content of the file",
						),
					},
					RunFn: func(
						_ context.Context,
						_ proxias.Session,
						params map[string]any,
					) (proxias.ActionExecutionResult, error) {
						file := path(params)
						err := os.MkdirAll(filepath.Dir(file), 0o750)
						if err == nil {
							err = os.WriteFile(file, []byte(fmt.Sprint(params["content"])), 0o600)
						}
						if err != nil {
							//nolint:nilerr // its result says why
							return proxias.ActionExecutionResult{
								ActionID: "write_file",
								Status:   "failed",
								Data:     map[string]any{"error": err.Error()},
							}, nil
						}
						return proxias.ActionExecutionResult{
							ActionID: "write_file",
							Status:   "completed",
							Data:     map[string]any{"path": file},
						}, nil
					},
				},
				"execute_bash": {
					ActionDescriptor: proxias.ActionDescriptor{
						ID: "execute_bash",
						Description: "Runs a bash command in the workspace, " +
							"for up to 60 seconds, and tells its output and " +
							"exit code.",
						InputSchema: agentdemo.Params("command", "the command, as bash -c runs it"),
					},
					RunFn: func(
						ctx context.Context,
						_ proxias.Session,
						params map[string]any,
					) (proxias.ActionExecutionResult, error) {
						ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
						defer cancel()
						var stdout, stderr strings.Builder
						cmd := exec.CommandContext(ctx, "bash", "-c", fmt.Sprint(params["command"]))
						cmd.Dir, cmd.Stdout, cmd.Stderr = dir, &stdout, &stderr
						status, code := "completed", 0
						if err := cmd.Run(); err != nil {
							status, code = "failed", -1
							//nolint:errorlint // Run returns it unwrapped
							if exit, ok := err.(*exec.ExitError); ok {
								code = exit.ExitCode()
							}
							if ctx.Err() != nil {
								stderr.WriteString("\n(timed out after 60s)")
							}
						}
						return proxias.ActionExecutionResult{
							ActionID: "execute_bash",
							Status:   status,
							Data: map[string]any{
								"stdout":    stdout.String(),
								"stderr":    stderr.String(),
								"exit_code": code,
							},
						}, nil
					},
				},
			},
		},
	}
	var llm proxias.LLM
	switch *vendor {
	case "anthropic":
		if *model == "" {
			*model = agentdemo.ClaudeModel
		}
		// Claude thinks adaptively, and says what it thought: a summary of its
		// thinking.
		claude := agentanthropic.NewLLM(agentdemo.Client(), *model)
		claude.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{
				Display: anthropic.ThinkingConfigAdaptiveDisplaySummarized,
			},
		}
		llm = claude
	case "openai":
		if *model == "" {
			*model = "gpt-5-mini"
		}
		llm = agentopenai.NewLLM(
			openaisdk.NewClient(),
			*model,
		) // its key: OPENAI_API_KEY
	default:
		fmt.Fprintf(os.Stderr, "-vendor %q: it is anthropic or openai\n", *vendor)
		os.Exit(2)
	}
	strategy := proxias.NewReActStrategy(llm)

	fmt.Println(
		"Talking to Cleo (" + *vendor + ", " + *model + ", in " + dir + "). " +
			"exit quits, Ctrl-C cancels a turn.",
	)
	session := proxias.NewSession(
		proxias.NewMessageEvent(
			proxias.NewSystemSource("cli"),
			"You can read and write files, and run bash commands, in your "+
				"workspace ("+dir+"). When you lack a capability, write "+
				"yourself a script that gives it to you, run it, and reuse "+
				"it later. Keep your memory current with update_memory: what "+
				"you learn about the user, and the scripts you wrote and "+
				"what they do.",
		),
	)
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
		stream, err := agent.Loop(
			ctx,
			session.WithEvents(proxias.NewMessageEvent(proxias.NewUserSource("you"), text)),
			strategy,
		)
		if err != nil {
			cancel()
			fmt.Fprintln(os.Stderr, err)
			return
		}
		// The events as they come (not the chunks: a text is a thought or the
		// reply only once its event says so).
		for update := range stream.Updates() {
			if update.Err != nil {
				fmt.Fprintln(os.Stderr, "error:", update.Err)
				continue
			}
			for _, e := range update.Events {
				switch {
				case e.Type == proxias.EventTypeThought:
					// The model's thinking is kept whole, as JSON: its text is
					// what to show.
					thought := e.Text()
					var thinking proxias.LLMChunk
					if json.Unmarshal(e.Payload, &thinking) == nil && thinking.Thinking != "" {
						thought = thinking.Thinking
					}
					if thought = strings.TrimSpace(thought); thought != "" {
						fmt.Printf("\033[2;3mthinking: %s\033[0m\n", thought)
					}
				case e.Type == proxias.EventTypeActionRequest || e.Type == proxias.EventTypeActionResult:
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
