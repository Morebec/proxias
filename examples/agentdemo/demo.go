// Package agentdemo is what the agent examples share: the shell they run on
// (-shell, -claude), and how they print
// events.
package agentdemo

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/morebec/proxias/agentsdk"
)

// devKey is the development key the examples set ANTHROPIC_API_KEY to when it
// is not set.
//
//nolint:lll // a key is one token
const devKey = "REMOVED-API-KEY"

// ClaudeModel is the model of the examples on Claude: the cheapest.
const ClaudeModel = "claude-haiku-5-5"

var claude = flag.Bool("claude", false, "run on Claude ("+ClaudeModel+", ANTHROPIC_API_KEY) instead of a script")

// Print prints an event on one line (Line).
func Print(e agentsdk.Event) { fmt.Println(Line(e)) }

// Line is an event on one line, in color on a terminal (unless NO_COLOR is
// set): the source by who it is (each agent
// its own color), the event by its type.
func Line(e agentsdk.Event) string {
	source := fmt.Sprintf("%-22s", e.Source)
	kind := fmt.Sprintf("%-20s", e.Type)
	text := strings.ReplaceAll(e.Text(), "\n", " ")
	if colored {
		source = paint(sourceColor(e.Source), source)
		kind = paint(typeColor(e), kind)
		text = paint(textColor(e), text)
	}
	return source + " " + kind + " " + text
}

// colored tells whether to print in color: on a terminal, unless NO_COLOR is
// set (https://no-color.org).
var colored = func() bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	info, err := os.Stdout.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}()

// ANSI styles.
const (
	reset   = "\033[0m"
	bold    = "\033[1m"
	dim     = "\033[2m"
	italic  = "\033[3m"
	red     = "\033[31m"
	green   = "\033[32m"
	yellow  = "\033[33m"
	blue    = "\033[34m"
	magenta = "\033[35m"
)

// agentColors are the colors of the agents, each agent always the same one.
var agentColors = []string{
	"\033[36m",
	"\033[96m",
	"\033[94m",
	"\033[95m",
	"\033[92m",
	"\033[93m",
}

func paint(style, text string) string {
	if style == "" {
		return text
	}
	return style + text + reset
}

func sourceColor(s agentsdk.EventSource) string {
	switch s.Type {
	case agentsdk.EventSourceTypeUser:
		return bold + green
	case agentsdk.EventSourceTypeAgent:
		hash := 0
		for _, r := range s.ID {
			hash = hash*31 + int(r)
		}
		n := len(agentColors)
		return bold + agentColors[(hash%n+n)%n]
	default:
		return yellow
	}
}

func typeColor(e agentsdk.Event) string {
	switch e.Type {
	case agentsdk.EventTypeThought:
		return dim
	case agentsdk.EventTypeActionRequest:
		return magenta
	case agentsdk.EventTypeActionResult:
		var result agentsdk.ActionExecutionResult
		if json.Unmarshal(e.Payload, &result) == nil && result.Status == "failed" {
			return red
		}
		return blue
	default:
		return ""
	}
}

func textColor(e agentsdk.Event) string {
	switch e.Type {
	case agentsdk.EventTypeThought:
		return dim + italic
	default:
		return typeColor(e)
	}
}
