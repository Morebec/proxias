package agentsdk

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
)

// DefaultSystem1SelectionThreshold is the probability under which a
// System1ActionSelector selects nothing, when its SelectionThreshold is nil:
// that the last message asks for an action, and the confidence of the choice
// of the action.
const DefaultSystem1SelectionThreshold = 0.5

// ActionClassifier decides which of the actions the last message of the
// invocation's session asks the agent to run now, and with which parameters:
// none when it asks for none.
type ActionClassifier interface {
	Classify(ctx context.Context, inv Invocation, descriptors []ActionDescriptor) ([]ActionExecutionRequest, error)
}

// LLMActionClassifier is an ActionClassifier relying on an LLM in a single
// call: the model reads the actions, with their schemas, and the messages, and
// answers
// with the executions and their parameters. It needs no System One model; a
// CompositeActionClassifier can replace it later.
type LLMActionClassifier struct {
	LLM LLM
	// Prompt is the system prompt: DefaultLLMClassificationPrompt when empty.
	Prompt string
}

// Classify asks the model. Without actions the model is not called.
func (c LLMActionClassifier) Classify(
	ctx context.Context,
	inv Invocation,
	descriptors []ActionDescriptor,
) ([]ActionExecutionRequest, error) {
	if len(descriptors) == 0 {
		return nil, nil
	}
	prompt := c.Prompt
	if prompt == "" {
		prompt = DefaultLLMClassificationPrompt
	}
	inventory, err := json.Marshal(map[string]any{"actions": descriptors})
	if err != nil {
		return nil, err
	}
	llmRequest := NewLLMRequestFromSession(inv.Agent.AsSource(), inv.Session)
	instructions := []LLMMessage{
		{
			Role:  LLMMessageRoleSystem,
			Parts: []LLMPart{{Text: prompt}},
		},
		{
			Role:  LLMMessageRoleSystem,
			Parts: []LLMPart{{Text: string(inventory)}},
		},
	}
	if len(inv.Memories) > 0 {
		remembered := "# What you remember"
		for _, memory := range inv.Memories {
			remembered += "\n- " + memory.Text()
		}
		instructions = append(
			instructions,
			LLMMessage{
				Role:  LLMMessageRoleSystem,
				Parts: []LLMPart{{Text: remembered}},
			},
		)
	}
	llmRequest.Messages = append(instructions, llmRequest.Messages...)
	var text strings.Builder
	for chunk, err := range c.LLM.Complete(ctx, llmRequest) {
		if err != nil {
			return nil, fmt.Errorf("classifying: %w", err)
		}
		text.WriteString(chunk.Text)
	}
	// The first JSON value of the answer: a code fence or words around it are
	// tolerated, braces in words after it too.
	answerText := text.String()
	start := strings.Index(answerText, "{")
	if start < 0 {
		return nil, errors.New("classifying: the model's answer has no JSON object")
	}
	var decoded struct {
		Executions []ActionExecutionRequest `json:"executions"`
	}
	decoder := json.NewDecoder(strings.NewReader(answerText[start:]))
	if err := decoder.Decode(&decoded); err != nil {
		return nil, fmt.Errorf("classifying: the model's answer is not the executions: %w", err)
	}
	return decoded.Executions, nil
}

// ActionSelector is Stage 1 of a CompositeActionClassifier: ideally a fast,
// System One triage, it chooses the actions the last message asks for, without
// extracting their parameters. None means that it asks for none.
type ActionSelector interface {
	Select(ctx context.Context, inv Invocation, descriptors []ActionDescriptor) ([]ActionDescriptor, error)
}

// ActionParameterExtractor is Stage 2 of a CompositeActionClassifier: it gives
// the executions of the selected actions, with the parameters the conversation
// gives them, as their schemas say. A parameter the conversation does not give
// is left out.
type ActionParameterExtractor interface {
	ExtractParameters(
		ctx context.Context,
		inv Invocation,
		selected []ActionDescriptor,
	) ([]ActionExecutionRequest, error)
}

// CompositeActionClassifier is the ActionClassifier of two stages, so that each
// can be a different component: the Selector chooses the actions, then the
// ParameterExtractor extracts the parameters of those, and of those only.
type CompositeActionClassifier struct {
	Selector           ActionSelector
	ParameterExtractor ActionParameterExtractor
}

// NewCompositeActionClassifier returns the CompositeActionClassifier of the
// selector and the parameter extractor.
func NewCompositeActionClassifier(
	selector ActionSelector,
	extractor ActionParameterExtractor,
) CompositeActionClassifier {
	return CompositeActionClassifier{
		Selector:           selector,
		ParameterExtractor: extractor,
	}
}

// Classify selects the actions, then extracts their parameters. Nothing is
// extracted when nothing was selected.
func (c CompositeActionClassifier) Classify(
	ctx context.Context,
	inv Invocation,
	descriptors []ActionDescriptor,
) ([]ActionExecutionRequest, error) {
	selected, err := c.Selector.Select(ctx, inv, descriptors)
	if err != nil {
		return nil, err
	}
	if len(selected) == 0 {
		return nil, nil // no action is asked for: nothing to extract
	}
	return c.ParameterExtractor.ExtractParameters(ctx, inv, selected)
}

// LLMActionSelector is an ActionSelector relying on an LLM: the model answers
// with the IDs of the actions the last message asks for. It does not see their
// schemas: it chooses by name and description only.
type LLMActionSelector struct {
	LLM LLM
	// Prompt is the system prompt: DefaultLLMSelectionPrompt when empty.
	Prompt string
}

// Select asks the model. It returns the actions of the IDs it answers, in its
// order: an ID that is not among the actions is an action of that ID only.
// Without actions the model is not called.
func (s LLMActionSelector) Select(
	ctx context.Context,
	inv Invocation,
	descriptors []ActionDescriptor,
) ([]ActionDescriptor, error) {
	if len(descriptors) == 0 {
		return nil, nil
	}
	prompt := s.Prompt
	if prompt == "" {
		prompt = DefaultLLMSelectionPrompt
	}
	unschemed := make([]ActionDescriptor, len(descriptors))
	for i, descriptor := range descriptors {
		descriptor.InputSchema = nil
		unschemed[i] = descriptor
	}
	inventory, err := json.Marshal(map[string]any{"actions": unschemed})
	if err != nil {
		return nil, err
	}
	llmRequest := NewLLMRequestFromSession(inv.Agent.AsSource(), inv.Session)
	instructions := []LLMMessage{
		{
			Role:  LLMMessageRoleSystem,
			Parts: []LLMPart{{Text: prompt}},
		},
		{
			Role:  LLMMessageRoleSystem,
			Parts: []LLMPart{{Text: string(inventory)}},
		},
	}
	if len(inv.Memories) > 0 {
		remembered := "# What you remember"
		for _, memory := range inv.Memories {
			remembered += "\n- " + memory.Text()
		}
		instructions = append(
			instructions,
			LLMMessage{
				Role:  LLMMessageRoleSystem,
				Parts: []LLMPart{{Text: remembered}},
			},
		)
	}
	llmRequest.Messages = append(instructions, llmRequest.Messages...)
	var text strings.Builder
	for chunk, err := range s.LLM.Complete(ctx, llmRequest) {
		if err != nil {
			return nil, fmt.Errorf("selecting actions: %w", err)
		}
		text.WriteString(chunk.Text)
	}
	// The first JSON value of the answer: a code fence or words around it are
	// tolerated, braces in words after it too.
	answerText := text.String()
	start := strings.Index(answerText, "{")
	if start < 0 {
		return nil, errors.New("selecting actions: the model's answer has no JSON object")
	}
	var decoded struct {
		Actions []string `json:"actions"`
	}
	decoder := json.NewDecoder(strings.NewReader(answerText[start:]))
	if err := decoder.Decode(&decoded); err != nil {
		return nil, fmt.Errorf("selecting actions: the model's answer is not the actions: %w", err)
	}
	selected := make([]ActionDescriptor, len(decoded.Actions))
	for i, id := range decoded.Actions {
		selected[i] = ActionDescriptor{ID: id}
		for _, descriptor := range descriptors {
			if descriptor.ID == id {
				selected[i] = descriptor
			}
		}
	}
	return selected, nil
}

// System1ActionSelector is an ActionSelector of a System1Model: concurrently, a
// noul (whether the last message asks for an action) and a choice (which). It
// selects the chosen action when both are likely enough.
type System1ActionSelector struct {
	System1 System1Model
	// SelectionThreshold defaults DefaultSystem1SelectionThreshold when nil; a
	// pointer to 0 selects whatever System1 chooses.
	SelectionThreshold *float64
	// ChoiceInstructions defaults DefaultSystem1SelectorChoiceInstructions
	ChoiceInstructions string
	// NoulInstructions defaults DefaultSystem1SelectorNoulInstructions
	NoulInstructions string
}

// Select asks System1 its two questions, concurrently. Without actions it is
// not asked.
func (s System1ActionSelector) Select(
	ctx context.Context,
	inv Invocation,
	descriptors []ActionDescriptor,
) ([]ActionDescriptor, error) {
	if len(descriptors) == 0 {
		return nil, nil
	}
	choiceInstructions := s.ChoiceInstructions
	if choiceInstructions == "" {
		choiceInstructions = DefaultSystem1SelectorChoiceInstructions
	}

	noulInstructions := s.NoulInstructions
	if noulInstructions == "" {
		noulInstructions = DefaultSystem1SelectorNoulInstructions
	}

	// The state is the session as the request of a model, as JSON: its
	// instructions and its messages.
	llmRequest := NewLLMRequestFromSession(inv.Agent.AsSource(), inv.Session)
	var instructions []LLMMessage
	if len(inv.Memories) > 0 {
		remembered := "# What you remember"
		for _, memory := range inv.Memories {
			remembered += "\n- " + memory.Text()
		}
		instructions = append(
			instructions,
			LLMMessage{
				Role:  LLMMessageRoleSystem,
				Parts: []LLMPart{{Text: remembered}},
			},
		)
	}
	llmRequest.Messages = append(instructions, llmRequest.Messages...)
	state, err := json.Marshal(llmRequest)
	if err != nil {
		return nil, err
	}
	criteria := map[string]string{}
	for _, descriptor := range descriptors {
		criteria[descriptor.ID] = descriptor.Description
		if descriptor.Name != "" && descriptor.Name != descriptor.ID {
			criteria[descriptor.ID] = descriptor.Name + ": " + descriptor.Description
		}
	}
	var (
		wait               sync.WaitGroup
		choice             System1ChoiceAnswer
		noul               System1NoulAnswer
		choiceErr, noulErr error
	)
	wait.Go(func() {
		choice, choiceErr = s.System1.AnswerChoice(
			ctx,
			string(state),
			System1Question{
				Instructions: choiceInstructions,
				Criteria:     criteria,
			},
		)
	})
	wait.Go(func() {
		noul, noulErr = s.System1.AnswerNoul(
			ctx,
			string(state),
			System1Question{Instructions: noulInstructions},
		)
	})
	wait.Wait()
	if err := errors.Join(choiceErr, noulErr); err != nil {
		return nil, fmt.Errorf("selecting actions: %w", err)
	}
	threshold := DefaultSystem1SelectionThreshold
	if s.SelectionThreshold != nil {
		threshold = *s.SelectionThreshold
	}
	if noul.Noul < threshold || choice.Confidence < threshold {
		return nil, nil
	}
	var selected []ActionDescriptor
	for _, descriptor := range descriptors {
		if descriptor.ID == choice.Choice {
			selected = append(selected, descriptor)
		}
	}
	return selected, nil
}

// LLMActionParameterExtractor is an ActionParameterExtractor of an LLM: the
// model answers with the executions of the selected actions, with the
// parameters
// the conversation gives them, as their schemas say.
type LLMActionParameterExtractor struct {
	LLM LLM
	// Prompt is the system prompt: DefaultLLMExtractionPrompt when empty.
	Prompt string
}

// ExtractParameters asks the model. Without actions the model is not called.
func (ex LLMActionParameterExtractor) ExtractParameters(
	ctx context.Context,
	inv Invocation,
	selected []ActionDescriptor,
) ([]ActionExecutionRequest, error) {
	if len(selected) == 0 {
		return nil, nil
	}
	prompt := ex.Prompt
	if prompt == "" {
		prompt = DefaultLLMExtractionPrompt
	}
	inventory, err := json.Marshal(map[string]any{"actions": selected})
	if err != nil {
		return nil, err
	}
	llmRequest := NewLLMRequestFromSession(inv.Agent.AsSource(), inv.Session)
	instructions := []LLMMessage{
		{
			Role:  LLMMessageRoleSystem,
			Parts: []LLMPart{{Text: prompt}},
		},
		{
			Role:  LLMMessageRoleSystem,
			Parts: []LLMPart{{Text: string(inventory)}},
		},
	}
	if len(inv.Memories) > 0 {
		remembered := "# What you remember"
		for _, memory := range inv.Memories {
			remembered += "\n- " + memory.Text()
		}
		instructions = append(
			instructions,
			LLMMessage{
				Role:  LLMMessageRoleSystem,
				Parts: []LLMPart{{Text: remembered}},
			},
		)
	}
	llmRequest.Messages = append(instructions, llmRequest.Messages...)

	var text strings.Builder
	for chunk, err := range ex.LLM.Complete(ctx, llmRequest) {
		if err != nil {
			return nil, fmt.Errorf("extracting parameters: %w", err)
		}
		text.WriteString(chunk.Text)
	}

	// The first JSON value of the answer: a code fence or words around it are
	// tolerated, braces in words after it too.
	answerText := text.String()
	start := strings.Index(answerText, "{")
	if start < 0 {
		return nil, errors.New("extracting parameters: the model's answer has no JSON object")
	}
	var decoded struct {
		Executions []ActionExecutionRequest `json:"executions"`
	}
	decoder := json.NewDecoder(strings.NewReader(answerText[start:]))
	if err := decoder.Decode(&decoded); err != nil {
		return nil, fmt.Errorf(
			"extracting parameters: "+
				"the model's answer is not the executions: %w",
			err,
		)
	}
	return decoded.Executions, nil
}
