package proxias

import "context"

// IntentClassifierFunc is an IntentClassifier as a function.
type IntentClassifierFunc func(context.Context, ChatCompletionRequest, []ActionDescriptor) (Intent, error)

// ActionExecutionIntentSelectorFunc is an ActionExecutionIntentSelector as a function.
type ActionExecutionIntentSelectorFunc func(context.Context, ChatCompletionRequest, []ActionDescriptor) ([]ActionDescriptor, error)

// ActionExecutionIntentParameterExtractorFunc is an ActionExecutionIntentParameterExtractor as a function.
type ActionExecutionIntentParameterExtractorFunc func(ctx context.Context, req ChatCompletionRequest, matchedDescs []ActionDescriptor) ([]ActionExecutionIntent, error)

// IntentClassifier translates an inbound chat completion request into a typed Intent.
// It serves as the primary inbound translation component evaluating human prose against specific action descriptors
// to determine the request requires code execution through actions or open-ended reasoning.
// Harnesses commonly leave this to an LLM in its ReAct loop, which is slow, costly and non-deterministic. An
// IntentClassifier is meant to be fast: local rules, or a discriminative, system-1 model.
//
// It is given the actions that the request may ask for, with their InputSchema, and returns an Intent that
// names the actions to run and their parameters. An Intent with no action executions means that the request
// asks for none: it goes to the ReasoningEngine.
//
// Error Handling: return a non-nil error if classification itself fails, these errors are internal infrastructure level errors
// and won't be synthesized into a user-facing response by the control plane.
type IntentClassifier interface {
	Classify(context.Context, ChatCompletionRequest, []ActionDescriptor) (Intent, error)
}

// Classify calls f.
func (f IntentClassifierFunc) Classify(ctx context.Context, req ChatCompletionRequest, descs []ActionDescriptor) (Intent, error) {
	return f(ctx, req, descs)
}

// Intent represents the classified outcome of an inbound chat completion request.
// It contains information about which actions should be executed and with what parameters.
// An Intent with no ActionExecutions indicates that the request requires open-ended reasoning rather than
// specific action execution: it is answered by the ReasoningEngine.
type Intent struct {
	ActionExecutions []ActionExecutionIntent
}

// IsReasoningIntent tells whether the request asks for no action, and requires open-ended reasoning.
func (i Intent) IsReasoningIntent() bool {
	return len(i.ActionExecutions) == 0
}

// ActionExecutionIntent asks to run an action: it is the command, or the query, that a request was turned
// into. The action is run by the ActionExecutionEngine with Params, and the synthesizer receives the
// result along with this intent.
//
// The control plane does not check that the ActionID is one of those described: an engine answers an
// unknown action with a failed ActionExecutionResult.
type ActionExecutionIntent struct {
	// ActionID is the ID of the action to run.
	ActionID string `json:"action_id"`
	// Params are the values of the action's parameters, by name, as its InputSchema describes them.
	Params map[string]any `json:"params,omitempty"`
}

// CompositeIntentClassifier is an IntentClassifier made of two stages, so that each can be a different,
// small and fast component: an ActionExecutionIntentSelector chooses the actions, and a
// ActionExecutionIntentParameterExtractor then extracts the parameters of those that were chosen, and of
// those only.
type CompositeIntentClassifier struct {
	selector       ActionExecutionIntentSelector
	paramExtractor ActionExecutionIntentParameterExtractor
}

// NewCompositeIntentClassifier makes a CompositeIntentClassifier of a selector and a parameter extractor.
func NewCompositeIntentClassifier(selector ActionExecutionIntentSelector, paramExtractor ActionExecutionIntentParameterExtractor) *CompositeIntentClassifier {
	return &CompositeIntentClassifier{
		selector:       selector,
		paramExtractor: paramExtractor,
	}
}

// Classify selects the actions the request asks for, then extracts their parameters. Nothing is extracted
// if no action was selected: the intent is then a reasoning intent.
func (c *CompositeIntentClassifier) Classify(ctx context.Context, req ChatCompletionRequest, descs []ActionDescriptor) (Intent, error) {
	matchedActionDescs, err := c.selector.Select(ctx, req, descs)
	if err != nil {
		return Intent{}, err
	}

	intent := Intent{}

	if len(matchedActionDescs) != 0 {
		intent.ActionExecutions, err = c.paramExtractor.ExtractParameters(ctx, req, matchedActionDescs)
		if err != nil {
			return Intent{}, err
		}
	}

	return intent, nil
}

// ActionExecutionIntentSelector performs ideally fast System-1 triage on an incoming chat completion
// request to determine the most appropriate actions.
// It acts as Stage 1 of a CompositeIntentClassifier, evaluating user prose to decide
// whether a chat completion request maps to specific action IDs or requires open-ended reasoning,
// without spending tokens on full parameter extraction.
// Returning an empty slice indicates that no suitable actions were found.
//
// Error Handling: return a non-nil error if action selection itself fails, these are internal infrastructure level errors
// and won't be synthesized into a user-facing response by the control plane.
type ActionExecutionIntentSelector interface {
	Select(context.Context, ChatCompletionRequest, []ActionDescriptor) ([]ActionDescriptor, error)
}

// Select implements ActionExecutionIntentSelector.
func (f ActionExecutionIntentSelectorFunc) Select(ctx context.Context, req ChatCompletionRequest, descs []ActionDescriptor) ([]ActionDescriptor, error) {
	return f(ctx, req, descs)
}

// ActionExecutionIntentParameterExtractor binds user prose to strongly typed action parameter schemas.
// It acts as Stage 2 of a CompositeIntentClassifier, taking the targeted action descriptors
// identified during Stage 1 selection and extracting the populated parameter maps.
// A parameter that the request does not give is left out of the map: the action, which knows what it
// requires, fails with a result that says what is missing.
//
// Error Handling: return a non-nil error if parameter extraction itself fails, these are internal infrastructure level errors
// and won't be synthesized into a user-facing response by the control plane.
type ActionExecutionIntentParameterExtractor interface {
	ExtractParameters(ctx context.Context, req ChatCompletionRequest, matchedDescs []ActionDescriptor) ([]ActionExecutionIntent, error)
}

// ExtractParameters implements ActionExecutionIntentParameterExtractor.
func (f ActionExecutionIntentParameterExtractorFunc) ExtractParameters(ctx context.Context, req ChatCompletionRequest, matchedDescs []ActionDescriptor) ([]ActionExecutionIntent, error) {
	return f(ctx, req, matchedDescs)
}
