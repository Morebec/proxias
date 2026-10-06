package rules

import (
	"context"
	"regexp"
	"strings"

	"github.com/morebec/proxias"
)

// Rule maps a message to a Workflow.
type Rule struct {
	Pattern    *regexp.Regexp // matched against the last message
	WorkflowID string
}

// Classifier is an IntentClassifier of ordered rules: the first matching Rule wins.
type Classifier struct{ rules []Rule }

// New makes a Classifier of rules.
func New(rules ...Rule) *Classifier { return &Classifier{rules} }

// Command is the Rule of a slash command: "/<command>" at the start of the message names the Workflow.
func Command(command, workflowID string) Rule {
	return Rule{Pattern: regexp.MustCompile(`^/` + regexp.QuoteMeta(command) + `(\s|$)`), WorkflowID: workflowID}
}

const runPrefix = "/run "

// RunCommand is an IntentClassifier for "/run <workflow id>" on the last line of the message, whatever
// Workflows are registered, an unknown id left for the Interceptor to ignore.
type RunCommand struct{}

func (RunCommand) Classify(_ context.Context, request proxias.Request, candidates []proxias.Candidate) (proxias.Intent, error) {
	last, _ := request.Last()
	lines := strings.Split(strings.TrimSpace(last.Content), "\n")
	line := strings.TrimSpace(lines[len(lines)-1])
	if !strings.HasPrefix(line, runPrefix) {
		return proxias.NoIntent, nil
	}
	id := strings.TrimSpace(strings.TrimPrefix(line, runPrefix))
	for _, candidate := range candidates {
		if candidate.ID == id {
			return proxias.Intent{WorkflowIDs: []string{id}}, nil
		}
	}
	return proxias.NoIntent, nil
}

func (c *Classifier) Classify(_ context.Context, request proxias.Request, _ []proxias.Candidate) (proxias.Intent, error) {
	last, _ := request.Last()
	for _, rule := range c.rules {
		if rule.Pattern.MatchString(strings.TrimSpace(last.Content)) {
			return proxias.Intent{WorkflowIDs: []string{rule.WorkflowID}}, nil
		}
	}
	return proxias.NoIntent, nil
}
