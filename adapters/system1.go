// Package adapters adapts models to proxias: LLMSystem1, an LLM answering as a
// System1Model; the vendors' LLMs are in
// its subpackages (anthropic, openai).
package adapters

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/morebec/proxias"
)

// LLMSystem1 is an LLM as a System1Model: it is told the prompt of the kind of
// question, then the state and the
// question; its answer is read as the answer.
type LLMSystem1 struct {
	LLM proxias.LLM
	// ChoicePrompt, ScorePrompt and NoulPrompt are the system prompts of the
	// kinds of questions: DefaultChoicePrompt,
	// DefaultScorePrompt and DefaultNoulPrompt when empty.
	ChoicePrompt, ScorePrompt, NoulPrompt string
}

// AnswerChoice asks the model the choice about the state.
func (m LLMSystem1) AnswerChoice(
	ctx context.Context,
	state string,
	q proxias.System1Question,
) (proxias.System1ChoiceAnswer, error) {
	prompt := m.ChoicePrompt
	if prompt == "" {
		prompt = DefaultChoicePrompt
	}
	question, err := json.Marshal(q)
	if err != nil {
		return proxias.System1ChoiceAnswer{}, err
	}
	var text strings.Builder
	for c, err := range m.LLM.Complete(ctx, proxias.LLMRequest{
		Messages: []proxias.LLMMessage{
			{
				Role: proxias.LLMMessageRoleSystem,
				Parts: []proxias.LLMPart{
					{Text: prompt},
				},
			},
			{
				Role: proxias.LLMMessageRoleUser,
				Parts: []proxias.LLMPart{
					{Text: "# The state\n\n" + state},
					{Text: "# The question\n\n" + string(question)},
				},
			},
		},
	}) {
		if err != nil {
			return proxias.System1ChoiceAnswer{}, err
		}
		text.WriteString(c.Text)
	}
	answer := text.String()
	// The first JSON value of the answer: a code fence or words around it are
	// tolerated, braces in words after it too.
	start := strings.Index(answer, "{")
	if start < 0 {
		return proxias.System1ChoiceAnswer{}, errors.New(
			"the classifier's answer has no JSON object",
		)
	}
	var choice proxias.System1ChoiceAnswer
	decoder := json.NewDecoder(strings.NewReader(answer[start:]))
	if err := decoder.Decode(&choice); err != nil {
		return proxias.System1ChoiceAnswer{}, fmt.Errorf(
			"the classifier's answer is not a choice: %w",
			err,
		)
	}
	return choice, nil
}

// AnswerScore asks the model the score of the state.
func (m LLMSystem1) AnswerScore(
	ctx context.Context,
	state string,
	q proxias.System1Question,
) (proxias.System1ScoreAnswer, error) {
	prompt := m.ScorePrompt
	if prompt == "" {
		prompt = DefaultScorePrompt
	}
	question, err := json.Marshal(q)
	if err != nil {
		return proxias.System1ScoreAnswer{}, err
	}
	var text strings.Builder
	for c, err := range m.LLM.Complete(ctx, proxias.LLMRequest{
		Messages: []proxias.LLMMessage{
			{
				Role: proxias.LLMMessageRoleSystem,
				Parts: []proxias.LLMPart{
					{Text: prompt},
				},
			},
			{
				Role: proxias.LLMMessageRoleUser,
				Parts: []proxias.LLMPart{
					{Text: "# The state\n\n" + state},
					{Text: "# The question\n\n" + string(question)},
				},
			},
		},
	}) {
		if err != nil {
			return proxias.System1ScoreAnswer{}, err
		}
		text.WriteString(c.Text)
	}
	answer := text.String()
	// The first JSON value of the answer: a code fence or words around it are
	// tolerated, braces in words after it too.
	start := strings.Index(answer, "{")
	if start < 0 {
		return proxias.System1ScoreAnswer{}, errors.New(
			"the classifier's answer has no JSON object",
		)
	}
	var score proxias.System1ScoreAnswer
	decoder := json.NewDecoder(strings.NewReader(answer[start:]))
	if err := decoder.Decode(&score); err != nil {
		return proxias.System1ScoreAnswer{}, fmt.Errorf(
			"the classifier's answer is not a score: %w",
			err,
		)
	}
	return score, nil
}

// AnswerNoul asks the model how likely the statement is true of the state.
func (m LLMSystem1) AnswerNoul(
	ctx context.Context,
	state string,
	q proxias.System1Question,
) (proxias.System1NoulAnswer, error) {
	prompt := m.NoulPrompt
	if prompt == "" {
		prompt = DefaultNoulPrompt
	}
	question, err := json.Marshal(q)
	if err != nil {
		return proxias.System1NoulAnswer{}, err
	}
	var text strings.Builder
	for c, err := range m.LLM.Complete(ctx, proxias.LLMRequest{
		Messages: []proxias.LLMMessage{
			{
				Role: proxias.LLMMessageRoleSystem,
				Parts: []proxias.LLMPart{
					{Text: prompt},
				},
			},
			{
				Role: proxias.LLMMessageRoleUser,
				Parts: []proxias.LLMPart{
					{Text: "# The state\n\n" + state},
					{Text: "# The question\n\n" + string(question)},
				},
			},
		},
	}) {
		if err != nil {
			return proxias.System1NoulAnswer{}, err
		}
		text.WriteString(c.Text)
	}
	answer := text.String()
	// The first JSON value of the answer: a code fence or words around it are
	// tolerated, braces in words after it too.
	start := strings.Index(answer, "{")
	if start < 0 {
		return proxias.System1NoulAnswer{}, errors.New(
			"the classifier's answer has no JSON object",
		)
	}
	var noul proxias.System1NoulAnswer
	decoder := json.NewDecoder(strings.NewReader(answer[start:]))
	if err := decoder.Decode(&noul); err != nil {
		return proxias.System1NoulAnswer{}, fmt.Errorf(
			"the classifier's answer is not a noul: %w",
			err,
		)
	}
	return noul, nil
}
