package agentsdk

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// MemoryProvider is an agent's memory beyond a session: what it recalls for a
// turn (in NewAgentPrompt), what it memorizes of one, and the actions the
// agent curates it with. What bears on a turn, what is worth keeping, and whose
// memory it is (the invocation's session, and the session the actions run on,
// say who the agent deals with) are its own decisions.
type MemoryProvider interface {
	// Recall returns the memories that bear on the invocation, as events; none
	// is fine.
	Recall(ctx context.Context, inv Invocation) ([]Event, error)

	// Memorize is given the events a turn added, and keeps what it wants of
	// them.
	Memorize(ctx context.Context, inv Invocation, events []Event) error

	// ActionExecutor runs the actions the agent curates the memory with,
	// offered
	// next to its own; nil for none.
	ActionExecutor() ActionExecutor
}

// actionsWithMemory are the actions of an agent and those of its memory: each
// runs on its own executor.
type actionsWithMemory struct {
	actions, memory ActionExecutor
}

// Descriptors describes the agent's actions, then the memory's.
func (e actionsWithMemory) Descriptors(ctx context.Context, s Session) ([]ActionDescriptor, error) {
	memory, err := e.memory.Descriptors(ctx, s)
	if err != nil {
		return nil, err
	}
	if e.actions == nil {
		return memory, nil
	}
	actions, err := e.actions.Descriptors(ctx, s)
	if err != nil {
		return nil, err
	}
	return append(actions, memory...), nil
}

// Execute runs the memory's actions on its executor, the others on the agent's,
// and returns their results in order.
func (e actionsWithMemory) Execute(
	ctx context.Context,
	s Session,
	requests []ActionExecutionRequest,
) ([]ActionExecutionResult, error) {
	descriptors, err := e.memory.Descriptors(ctx, s)
	if err != nil {
		return nil, err
	}
	ofMemory := map[string]bool{}
	for _, d := range descriptors {
		ofMemory[d.ID] = true
	}
	// The requests of each executor, and where each is among the requests. An
	// agent without actions of its own has the
	// memory's executor answer them all.
	var memoryRequests, agentRequests []ActionExecutionRequest
	var memoryAt, agentAt []int
	for i, req := range requests {
		if ofMemory[req.ActionID] || e.actions == nil {
			memoryRequests = append(memoryRequests, req)
			memoryAt = append(memoryAt, i)
		} else {
			agentRequests = append(agentRequests, req)
			agentAt = append(agentAt, i)
		}
	}

	results := make([]ActionExecutionResult, len(requests))
	if len(memoryRequests) > 0 {
		executed, err := e.memory.Execute(ctx, s, memoryRequests)
		if err == nil && len(executed) != len(memoryRequests) {
			err = fmt.Errorf("%d results for %d actions", len(executed), len(memoryRequests))
		}
		if err != nil {
			return nil, err
		}
		for j, i := range memoryAt {
			results[i] = executed[j]
		}
	}
	if len(agentRequests) > 0 {
		executed, err := e.actions.Execute(ctx, s, agentRequests)
		if err == nil && len(executed) != len(agentRequests) {
			err = fmt.Errorf("%d results for %d actions", len(executed), len(agentRequests))
		}
		if err != nil {
			return nil, err
		}
		for j, i := range agentAt {
			results[i] = executed[j]
		}
	}
	return results, nil
}

// MarkdownMemory is a memory the agent curates itself, in one markdown document
// (a MEMORY.md): it is recalled whole at every turn, and the agent rewrites it
// with its action, update_memory. Nothing is memorized otherwise.
type MarkdownMemory struct {
	// Path is the document's file, created by the agent's first update.
	Path string
	// UpdateDescription is the description of update_memory:
	// DefaultMarkdownMemoryUpdateDescription when empty.
	UpdateDescription string
}

// Recall returns the document, as one memory; nothing when it is missing or
// empty.
func (m MarkdownMemory) Recall(context.Context, Invocation) ([]Event, error) {
	data, err := os.ReadFile(m.Path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(string(data)) == "" {
		return nil, nil
	}
	return []Event{
		NewMessageEvent(NewSystemSource("memory"), string(data)),
	}, nil
}

// Memorize keeps nothing: the agent curates the document.
func (m MarkdownMemory) Memorize(context.Context, Invocation, []Event) error {
	return nil
}

// ActionExecutor runs update_memory, which rewrites the document, creating its
// directory if needed.
func (m MarkdownMemory) ActionExecutor() ActionExecutor {
	description := m.UpdateDescription

	if description == "" {
		description = DefaultMarkdownMemoryUpdateDescription
	}

	return SimpleActionExecutor{
		Actions: map[string]SimpleAction{
			"update_memory": {
				ActionDescriptor: ActionDescriptor{
					ID:          "update_memory",
					Name:        "Update memory",
					Description: description,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"content": map[string]any{
								"type": "string",
								"description": "the whole new document, " +
									"in markdown",
							}},
						"required": []string{"content"},
					},
				},
				RunFn: func(_ context.Context, _ Session, params map[string]any) (ActionExecutionResult, error) {
					// A failure is the action's, told in its result.
					err := os.MkdirAll(filepath.Dir(m.Path), 0o750)
					if err != nil {
						//nolint:nilerr // its result says why
						return ActionExecutionResult{
							ActionID: "update_memory",
							Status:   "failed",
							Data:     map[string]any{"error": err.Error()},
						}, nil
					}
					err = os.WriteFile(m.Path, []byte(fmt.Sprint(params["content"])), 0o600)
					if err != nil {
						//nolint:nilerr // its result says why
						return ActionExecutionResult{
							ActionID: "update_memory",
							Status:   "failed",
							Data:     map[string]any{"error": err.Error()},
						}, nil
					}
					return ActionExecutionResult{
						ActionID: "update_memory",
						Status:   "completed",
					}, nil
				},
			},
		},
	}
}
