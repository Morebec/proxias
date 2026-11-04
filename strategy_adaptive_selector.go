package proxias

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// DefaultSystem1SelectionThreshold is the confidence under which a
// System1StrategySelector chooses no strategy, when its SelectionThreshold is
// nil.
const DefaultSystem1SelectionThreshold = 0.5

// StrategySelector chooses the strategy that suits the task the session's last
// outside event asks for, among the candidates: its name, or "" when none
// clearly does.
type StrategySelector interface {
	Select(ctx context.Context, inv Invocation, candidates []StrategyCandidate) (string, error)
}

// LLMStrategySelector is a StrategySelector relying on an LLM: the model reads
// the candidates' names and descriptions, the agent's actions (their names and
// descriptions) and the conversation, and answers the name of one.
type LLMStrategySelector struct {
	LLM LLM
	// Prompt is the system prompt: DefaultLLMStrategySelectionPrompt when empty.
	Prompt string
}

// Select asks the model.
func (s LLMStrategySelector) Select(
	ctx context.Context,
	inv Invocation,
	candidates []StrategyCandidate,
) (string, error) {
	prompt := s.Prompt
	if prompt == "" {
		prompt = DefaultLLMStrategySelectionPrompt
	}
	agentPrompt, err := NewAgentPrompt(ctx, inv)
	if err != nil {
		return "", err
	}
	options, err := strategyOptions(ctx, inv, candidates)
	if err != nil {
		return "", err
	}
	systemInstructions := LLMMessage{
		Role: LLMMessageRoleSystem,
		Parts: []LLMPart{
			{Text: prompt},
			{Text: agentPrompt.String()},
			{Text: options},
		},
	}
	llmRequest := NewLLMRequestFromSession(inv.Agent.AsSource(), inv.Session)
	llmRequest.Messages = slices.Insert(llmRequest.Messages, 0, systemInstructions)

	var text strings.Builder
	for chunk, err := range s.LLM.Complete(ctx, llmRequest) {
		if err != nil {
			return "", fmt.Errorf("selecting a strategy: %w", err)
		}
		text.WriteString(chunk.Text)
	}

	// The first JSON value of the answer: a code fence or words around it are
	// left out.
	answerText := text.String()
	start := strings.Index(answerText, "{")
	if start < 0 {
		return "", errors.New("selecting a strategy: the model's answer has no JSON object")
	}
	var decoded struct {
		Strategy string `json:"strategy"`
	}
	decoder := json.NewDecoder(strings.NewReader(answerText[start:]))
	if err := decoder.Decode(&decoded); err != nil {
		return "", fmt.Errorf("selecting a strategy: the model's answer is not a strategy: %w", err)
	}
	return decoded.Strategy, nil
}

// strategyOptions is what a selector is told to choose from, as JSON: the
// candidates, by name and description, and the agent's actions, by ID, name
// and description, without their schemas.
func strategyOptions(ctx context.Context, inv Invocation, candidates []StrategyCandidate) (string, error) {
	type option struct {
		Name        string `json:"name"`
		Description string `json:"description"`
	}
	strategies := make([]option, 0, len(candidates))
	for _, candidate := range candidates {
		strategies = append(strategies, option{Name: candidate.Name, Description: candidate.Description})
	}

	var actions []ActionDescriptor
	if inv.Agent.ActionExecutor != nil {
		descriptors, err := inv.Agent.ActionExecutor.Descriptors(ctx, inv.Session)
		if err != nil {
			return "", err
		}
		for _, descriptor := range descriptors {
			descriptor.InputSchema = nil
			actions = append(actions, descriptor)
		}
	}

	options, err := json.Marshal(map[string]any{"strategies": strategies, "actions": actions})
	if err != nil {
		return "", err
	}
	return string(options), nil
}

// System1StrategySelector is a StrategySelector of a System1Model: a choice
// among the candidates, by their descriptions, which it keeps when System1 is
// confident enough.
type System1StrategySelector struct {
	System1 System1Model
	// SelectionThreshold defaults DefaultSystem1SelectionThreshold when nil; a
	// pointer to 0 keeps whatever System1 chooses.
	SelectionThreshold *float64
	// ChoiceInstructions defaults DefaultSystem1StrategyChoiceInstructions.
	ChoiceInstructions string
}

// Select asks System1 its choice. Under the threshold, it chooses none.
func (s System1StrategySelector) Select(
	ctx context.Context,
	inv Invocation,
	candidates []StrategyCandidate,
) (string, error) {
	choiceInstructions := s.ChoiceInstructions
	if choiceInstructions == "" {
		choiceInstructions = DefaultSystem1StrategyChoiceInstructions
	}

	// The state is the session as the request of a model, as JSON: who the
	// agent is, its actions, and its messages.
	agentPrompt, err := NewAgentPrompt(ctx, inv)
	if err != nil {
		return "", err
	}
	options, err := strategyOptions(ctx, inv, nil)
	if err != nil {
		return "", err
	}
	agentInstructions := LLMMessage{
		Role: LLMMessageRoleSystem,
		Parts: []LLMPart{
			{Text: agentPrompt.String()},
			{Text: options},
		},
	}
	llmRequest := NewLLMRequestFromSession(inv.Agent.AsSource(), inv.Session)
	llmRequest.Messages = slices.Insert(llmRequest.Messages, 0, agentInstructions)
	state, err := json.Marshal(llmRequest)
	if err != nil {
		return "", err
	}

	criteria := map[string]string{}
	for _, candidate := range candidates {
		criteria[candidate.Name] = candidate.Description
	}
	choice, err := s.System1.AnswerChoice(
		ctx,
		string(state),
		System1Question{
			Instructions: choiceInstructions,
			Criteria:     criteria,
		},
	)
	if err != nil {
		return "", fmt.Errorf("selecting a strategy: %w", err)
	}

	threshold := DefaultSystem1SelectionThreshold
	if s.SelectionThreshold != nil {
		threshold = *s.SelectionThreshold
	}
	if choice.Confidence < threshold {
		return "", nil
	}
	return choice.Choice, nil
}
