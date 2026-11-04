package main

import (
	"fmt"
	"strings"

	"github.com/morebec/proxias"
)

// catalogScenarios are the scenarios of the examples (chat, unattended) on an
// agent with the hundred actions of the catalog. Each is strict: the calls
// made, rejected ones included, must be exactly the ones expected, with the
// right
// parameters, in order.
func catalogScenarios() []scenario {
	return []scenario{
		{name: "chat", run: catalogChat},
		{name: "unattended", run: catalogUnattended},
	}
}

// withCatalog is the agent, with the catalog's actions recorded in c.
func withCatalog(id string, persona proxias.AgentPersona, c *calls) proxias.Agent {
	return proxias.Agent{
		ID:             id,
		AgentPersona:   persona,
		ActionExecutor: catalogActions(c),
	}
}

// expected is a call expected, and how to tell its parameters are right.
type expected struct {
	action string
	right  func(params map[string]any) bool
}

// check tells whether the calls made are exactly the expected ones, and why
// not.
func check(made []call, want ...expected) (bool, string) {
	describe := func() string {
		s := make([]string, 0, len(made))
		for _, c := range made {
			r := ""
			if c.rejected {
				r = " (rejected)"
			}
			s = append(s, fmt.Sprintf("%s%v%s", c.action, c.params, r))
		}
		return "[" + strings.Join(s, ", ") + "]"
	}
	if len(made) != len(want) {
		return false, fmt.Sprintf("%d calls, %d expected: %s", len(made), len(want), describe())
	}
	for i, w := range want {
		if made[i].action != w.action {
			return false, fmt.Sprintf("call %d is %s, %s expected: %s", i+1, made[i].action, w.action, describe())
		}
		if made[i].rejected || !w.right(made[i].params) {
			return false, fmt.Sprintf("wrong parameters for %s: %s", w.action, describe())
		}
	}
	return true, ""
}

// has tells whether the parameter's text contains the part, whatever the case.
func has(params map[string]any, name, part string) bool {
	return strings.Contains(strings.ToLower(fmt.Sprint(params[name])), strings.ToLower(part))
}
