package proxias

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// Interceptor coordinates the IntentClassifier, the Workflows and the Fallback. It is safe for concurrent use.
type Interceptor struct {
	mu         sync.RWMutex
	workflows  map[string]Workflow
	order      []string
	classifier IntentClassifier
	fallback   Fallback
	hooks      Hooks
	logger     *slog.Logger
	input      func(Request) []byte
}

// NewInterceptor makes an Interceptor with no Workflow.
func NewInterceptor(options ...Option) *Interceptor {
	i := &Interceptor{workflows: map[string]Workflow{}, logger: slog.Default(), input: defaultInput}
	for _, option := range options {
		option(i)
	}
	return i
}

func defaultInput(request Request) []byte {
	last, _ := request.Last()
	input, _ := json.Marshal(map[string]string{"message": last.Content})
	return input
}

// Register adds a Workflow, replacing one of the same ID.
func (i *Interceptor) Register(workflow Workflow) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if _, exists := i.workflows[workflow.ID()]; !exists {
		i.order = append(i.order, workflow.ID())
	}
	i.workflows[workflow.ID()] = workflow
}

// Candidates are the registered Workflows, in registration order.
func (i *Interceptor) Candidates() []Candidate {
	i.mu.RLock()
	defer i.mu.RUnlock()
	candidates := make([]Candidate, 0, len(i.order))
	for _, id := range i.order {
		w := i.workflows[id]
		candidates = append(candidates, Candidate{ID: id, Name: w.Name(), Description: w.Description()})
	}
	return candidates
}

// Intercept answers a request. Only a last message from the user is classified: after a tool result the
// host continues its own tool loop, and an empty conversation asks nothing, so both go to the Fallback.
// A classifier that fails is as if it found no Intent: it never blocks an answer. The Workflows of the
// Intent run in order; one that fails ends the series, its failure being its Result. When the first one is
// not registered the message is handled as if it had no Intent. Then, unless Workflows ran and there is
// no Fallback (their Results are the answer), the Fallback answers from their Results.
func (i *Interceptor) Intercept(ctx context.Context, request Request) (Outcome, error) {
	results := i.run(ctx, request, i.classify(ctx, request))
	if i.fallback == nil {
		if len(results) == 0 {
			return Outcome{}, ErrNoFallback
		}
		return Outcome{Results: results}, nil
	}
	completion, err := i.fallback.Complete(ctx, request, results)
	if err != nil {
		return Outcome{Results: results}, err
	}
	return Outcome{Results: results, Completion: completion, FellBack: true}, nil
}

func (i *Interceptor) classify(ctx context.Context, request Request) Intent {
	if i.classifier == nil {
		return NoIntent
	}
	if last, ok := request.Last(); !ok || last.Role != RoleUser {
		return NoIntent
	}
	start := time.Now()
	intent, err := i.classifier.Classify(ctx, request, i.Candidates())
	if err != nil {
		i.logger.WarnContext(ctx, "message not classified, the fallback answers", slog.Any("error", err))
		intent = NoIntent
	}
	if i.hooks.Classified != nil {
		i.hooks.Classified(ctx, request, intent, err, time.Since(start).Nanoseconds())
	}
	return intent
}

func (i *Interceptor) run(ctx context.Context, request Request, intent Intent) []Result {
	if intent.IsNone() {
		return nil
	}
	input := intent.Input
	if input == nil {
		input = i.input(request)
	}
	var results []Result
	for _, id := range intent.WorkflowIDs {
		i.mu.RLock()
		workflow, ok := i.workflows[id]
		i.mu.RUnlock()
		if !ok {
			if len(results) == 0 {
				return nil
			}
			results = append(results, Result{WorkflowID: id, Failed: true, Output: failure(fmt.Errorf("workflow %s is not registered", id))})
			break
		}
		start := time.Now()
		result, err := workflow.Run(ctx, input)
		result.WorkflowID = id
		if err != nil {
			result.Failed, result.Output = true, failure(err)
		}
		if i.hooks.Ran != nil {
			i.hooks.Ran(ctx, result, err, time.Since(start).Nanoseconds())
		}
		results = append(results, result)
		if err != nil {
			break
		}
	}
	return results
}

// failure is the Output of a Workflow that failed.
func failure(err error) json.RawMessage {
	output, _ := json.Marshal(map[string]string{"error": err.Error()})
	return output
}
