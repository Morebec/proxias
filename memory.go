package proxias

import (
	"context"
	"errors"
	"os"
)

// MemoryProvider is an agent's memory beyond a session: what it remembers for
// a turn, as it is. The agent tells it in its prompt (NewAgentPrompt). How it
// is kept, and the actions that curate it, are the app's.
type MemoryProvider interface {
	// Recall returns what the agent remembers for the invocation, "" for
	// nothing. The invocation's session says who the agent deals with.
	Recall(ctx context.Context, inv Invocation) (string, error)
}

// MarkdownMemory is a memory in one markdown document (a MEMORY.md), recalled
// whole at every turn.
type MarkdownMemory struct {
	// Path is the document's file.
	Path string
}

// Recall returns the document, "" when it is missing.
func (m MarkdownMemory) Recall(context.Context, Invocation) (string, error) {
	data, err := os.ReadFile(m.Path)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	return string(data), err
}
