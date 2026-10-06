package proxias

import "context"

// IntentClassifier names what the last message of a request asks for, among the registered Workflows. It
// never runs anything. It may be a local rule or a remote model: only the Interceptor's own overhead is
// bounded by Proxias, not a classifier that calls a network.
type IntentClassifier interface {
	Classify(ctx context.Context, request Request, candidates []Candidate) (Intent, error)
}

// Intent is the Workflows to run, in order, for a message. The zero Intent (NoIntent) asks for none.
type Intent struct {
	WorkflowIDs []string
	// Input is given to every Workflow of the Intent. Nil: the Interceptor's default, below.
	Input []byte
}

// NoIntent is the Intent of a message that asks for no Workflow: it goes to the Fallback.
var NoIntent = Intent{}

// IsNone reports whether the Intent asks for no Workflow.
func (i Intent) IsNone() bool { return len(i.WorkflowIDs) == 0 }
