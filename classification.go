package proxias

import "context"

// IntentClassifier translates an inbound chat completion request into a typed Intent.
// It serves as the primary inbound translation component evaluating human prose against specific workflow descriptors
// to determine the request requires code execution through workflows or open-ended reasoning.
// Harnesses commonly leave this to an LLM in its ReAct loop, which is slow, costly and non-deterministic. An
// IntentClassifier is meant to be fast: local rules, or a discriminative, system-1 model.
//
// It is given the workflows that the request may ask for, with their InputSchema, and returns an Intent that
// names the workflows to run and their parameters. An Intent with no workflow executions means that the request
// asks for none: it goes to the ReasoningEngine.
//
// Error Handling: return a non-nil error if classification itself fails, these errors are internal infrastructure level errors
// and won't be synthesized into a user-facing response by the control plane.
type IntentClassifier interface {
	Classify(context.Context, CompletionRequest, []WorkflowDescriptor) (Intent, error)
}

// IntentClassifierFunc is an IntentClassifier as a function.
type IntentClassifierFunc func(context.Context, CompletionRequest, []WorkflowDescriptor) (Intent, error)

// Classify calls f.
func (f IntentClassifierFunc) Classify(ctx context.Context, req CompletionRequest, descs []WorkflowDescriptor) (Intent, error) {
	return f(ctx, req, descs)
}

// Intent represents the classified outcome of an inbound chat completion request.
// It contains information about which workflows should be executed and with what parameters.
// An Intent with no WorkflowExecutions indicates that the request requires open-ended reasoning rather than
// specific workflow execution: it is answered by the ReasoningEngine.
type Intent struct {
	WorkflowExecutions []WorkflowExecutionIntent
}

// IsReasoningIntent tells whether the request asks for no workflow, and requires open-ended reasoning.
func (i Intent) IsReasoningIntent() bool {
	return len(i.WorkflowExecutions) == 0
}

// WorkflowExecutionIntent asks to run a workflow: it is the command, or the query, that a request was turned
// into. The workflow is run by the WorkflowExecutionEngine with Params, and the synthesizer receives the
// result along with this intent.
//
// The control plane does not check that the WorkflowID is one of those described: an engine answers an
// unknown workflow with a failed WorkflowExecutionResult.
type WorkflowExecutionIntent struct {
	// WorkflowID is the ID of the workflow to run.
	WorkflowID string
	// Params are the values of the workflow's parameters, by name, as its InputSchema describes them.
	Params map[string]any
}

// CompositeIntentClassifier is an IntentClassifier made of two stages, so that each can be a different,
// small and fast component: a WorkflowExecutionIntentSelector chooses the workflows, and a
// WorkflowExecutionIntentParameterExtractor then extracts the parameters of those that were chosen, and of
// those only.
type CompositeIntentClassifier struct {
	selector       WorkflowExecutionIntentSelector
	paramExtractor WorkflowExecutionIntentParameterExtractor
}

// NewCompositeIntentClassifier makes a CompositeIntentClassifier of a selector and a parameter extractor.
func NewCompositeIntentClassifier(selector WorkflowExecutionIntentSelector, paramExtractor WorkflowExecutionIntentParameterExtractor) *CompositeIntentClassifier {
	return &CompositeIntentClassifier{
		selector:       selector,
		paramExtractor: paramExtractor,
	}
}

// Classify selects the workflows the request asks for, then extracts their parameters. Nothing is extracted
// if no workflow was selected: the intent is then a reasoning intent.
func (c CompositeIntentClassifier) Classify(ctx context.Context, req CompletionRequest, descs []WorkflowDescriptor) (Intent, error) {
	matchedWorkflowDescs, err := c.selector.Select(ctx, req, descs)
	if err != nil {
		return Intent{}, err
	}

	intent := Intent{}

	if len(matchedWorkflowDescs) != 0 {
		intent.WorkflowExecutions, err = c.paramExtractor.ExtractParameters(ctx, req, matchedWorkflowDescs)
		if err != nil {
			return Intent{}, err
		}
	}

	return intent, nil
}

// WorkflowExecutionIntentSelector performs ideally fast System-1 triage on an incoming chat completion
// request to determine the most appropriate workflows.
// It acts as Stage 1 of a CompositeIntentClassifier, evaluating user prose to decide
// whether a chat completion request maps to specific workflow IDs or requires open-ended reasoning,
// without spending tokens on full parameter extraction.
// Returning an empty slice indicates that no suitable workflows were found.
//
// Error Handling: return a non-nil error if workflow selection itself fails, these are internal infrastructure level errors
// and won't be synthesized into a user-facing response by the control plane.
type WorkflowExecutionIntentSelector interface {
	Select(context.Context, CompletionRequest, []WorkflowDescriptor) ([]WorkflowDescriptor, error)
}

// WorkflowExecutionIntentSelectorFunc is a WorkflowExecutionIntentSelector as a function.
type WorkflowExecutionIntentSelectorFunc func(context.Context, CompletionRequest, []WorkflowDescriptor) ([]WorkflowDescriptor, error)

// Select implements WorkflowExecutionIntentSelector.
func (f WorkflowExecutionIntentSelectorFunc) Select(ctx context.Context, req CompletionRequest, descs []WorkflowDescriptor) ([]WorkflowDescriptor, error) {
	return f(ctx, req, descs)
}

// WorkflowExecutionIntentParameterExtractor binds user prose to strongly typed workflow parameter schemas.
// It acts as Stage 2 of a CompositeIntentClassifier, taking the targeted workflow descriptors
// identified during Stage 1 selection and extracting the populated parameter maps.
// A parameter that the request does not give is left out of the map: the workflow, which knows what it
// requires, fails with a result that says what is missing.
//
// Error Handling: return a non-nil error if parameter extraction itself fails, these are internal infrastructure level errors
// and won't be synthesized into a user-facing response by the control plane.
type WorkflowExecutionIntentParameterExtractor interface {
	ExtractParameters(ctx context.Context, req CompletionRequest, matchedDescs []WorkflowDescriptor) ([]WorkflowExecutionIntent, error)
}

// WorkflowExecutionIntentParameterExtractorFunc is a WorkflowExecutionIntentParameterExtractor as a function.
type WorkflowExecutionIntentParameterExtractorFunc func(ctx context.Context, req CompletionRequest, matchedDescs []WorkflowDescriptor) ([]WorkflowExecutionIntent, error)

// ExtractParameters implements WorkflowExecutionIntentParameterExtractor.
func (f WorkflowExecutionIntentParameterExtractorFunc) ExtractParameters(ctx context.Context, req CompletionRequest, matchedDescs []WorkflowDescriptor) ([]WorkflowExecutionIntent, error) {
	return f(ctx, req, matchedDescs)
}
