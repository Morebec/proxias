package openai

import (
	"encoding/json"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"

	"github.com/morebec/proxias"
)

const (
	// describeActionsName is the tool that gives the model the schemas of
	// deferred actions. The adapter answers it, and asks the model again.
	describeActionsName = "describe_actions"
	// runActionName is the tool that runs a deferred action: the API has no
	// tool the model did not get up front.
	runActionName = "run_action"
	// maxDescribeRounds is how many times in one answer the model may ask for
	// schemas before it must act or answer.
	maxDescribeRounds = 3
)

// deferredActions are the actions the model gets by name only, in an index,
// and loads with describe_actions, by ID.
type deferredActions map[string]proxias.ActionDescriptor

// splitDeferred is the actions the model gets up front, and the deferred
// ones.
func splitDeferred(actions []proxias.ActionDescriptor) ([]proxias.ActionDescriptor, deferredActions) {
	var loaded []proxias.ActionDescriptor
	deferred := deferredActions{}
	for _, action := range actions {
		if action.Deferred {
			deferred[action.ID] = action
			continue
		}
		loaded = append(loaded, action)
	}
	return loaded, deferred
}

// tools are describe_actions and run_action.
func (d deferredActions) tools() []openai.ChatCompletionToolUnionParam {
	describe := shared.FunctionDefinitionParam{
		Name:        describeActionsName,
		Description: openai.String("Gives the parameters of actions to look up, by name, before you run them."),
		Parameters: shared.FunctionParameters{
			"type": "object",
			"properties": map[string]any{
				"names": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			},
			"required": []string{"names"},
		},
	}
	run := shared.FunctionDefinitionParam{
		Name:        runActionName,
		Description: openai.String("Runs an action to look up, described first, with its parameters."),
		Parameters: shared.FunctionParameters{
			"type": "object",
			"properties": map[string]any{
				"name":   map[string]any{"type": "string"},
				"params": map[string]any{"type": "object"},
			},
			"required": []string{"name", "params"},
		},
	}
	return []openai.ChatCompletionToolUnionParam{
		openai.ChatCompletionFunctionTool(describe),
		openai.ChatCompletionFunctionTool(run),
	}
}

// index lists the deferred actions, in the order the request gave them, for
// the system prompt.
func (d deferredActions) index(actions []proxias.ActionDescriptor) string {
	prompt := proxias.NewPrompt().AddSection(
		"Actions to look up",
		"Before you run one of these actions, get its parameters with "+describeActionsName+
			" (several at once if you need them), then run it with "+runActionName+".",
	)
	for _, action := range actions {
		if action.Deferred {
			line, _, _ := strings.Cut(strings.TrimSpace(action.Description), "\n")
			prompt.AddLabeledBulletPoint(action.ID, line)
		}
	}
	return prompt.String()
}

// describe is the schemas of the named actions, as JSON, for the model.
func (d deferredActions) describe(names []string) string {
	type description struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  map[string]any `json:"parameters"`
	}
	var unknown []string
	descriptions := []description{}
	for _, name := range names {
		action, ok := d[name]
		if !ok {
			unknown = append(unknown, name)
			continue
		}
		descriptions = append(descriptions, description{
			Name:        action.ID,
			Description: action.Description,
			Parameters:  action.InputSchema,
		})
	}
	answer := map[string]any{"actions": descriptions}
	if len(unknown) > 0 {
		answer["unknown"] = unknown
	}
	text, _ := json.Marshal(answer) // a schema came from JSON
	return string(text)
}

// describedActions is a describe_actions call and its answer, kept in the
// session as an opaque block, and given back as the call and its tool message.
type describedActions struct {
	CallID string   `json:"call_id"`
	Names  []string `json:"names"`
	Answer string   `json:"answer"`
}

// opaqueDescribed is the key of the opaque block of a describedActions: an
// opaque block without it is another adapter's.
const opaqueDescribed = "openai_described_actions"

func (d describedActions) opaque() json.RawMessage {
	block, _ := json.Marshal(map[string]describedActions{opaqueDescribed: d}) // strings marshal
	return block
}

// describedActionsFrom is the describe_actions exchange of an opaque block,
// and false for another adapter's block.
func describedActionsFrom(opaque json.RawMessage) (describedActions, bool) {
	var block map[string]describedActions
	if err := json.Unmarshal(opaque, &block); err != nil {
		return describedActions{}, false
	}
	d, ok := block[opaqueDescribed]
	return d, ok
}

// namesOf is the names a describe_actions call asks for.
func namesOf(input map[string]any) []string {
	raw, _ := input["names"].([]any)
	names := make([]string, 0, len(raw))
	for _, name := range raw {
		if s, ok := name.(string); ok {
			names = append(names, s)
		}
	}
	return names
}

// actionCallOf is a run_action call as the call of its action.
func actionCallOf(call *proxias.LLMToolCall) *proxias.LLMToolCall {
	name, _ := call.Input["name"].(string)
	params, _ := call.Input["params"].(map[string]any)
	return &proxias.LLMToolCall{ID: call.ID, Name: name, Input: params}
}

// runActionCallOf is the call of a deferred action as a run_action call, as
// the model made it.
func runActionCallOf(call *proxias.LLMToolCall) *proxias.LLMToolCall {
	params := call.Input
	if params == nil {
		params = map[string]any{}
	}
	return &proxias.LLMToolCall{
		ID:    call.ID,
		Name:  runActionName,
		Input: map[string]any{"name": call.Name, "params": params},
	}
}
