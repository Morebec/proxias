package agentsdk

import (
	"context"
	"iter"
)

// Role is who says a message of a conversation with a model.
type Role string

const (
	// RoleUser is everything that happened to the agent.
	RoleUser Role = "user"
	// RoleAssistant is the agent.
	RoleAssistant Role = "assistant"
)

// LLM is a language model, as every strategy sees it: given a system prompt, a conversation and tools, it streams
// text and tool calls. A vendor adapter implements it (anthropic.NewLLM); a script implements it for tests.
type LLM interface {
	Complete(ctx context.Context, req LLMRequest) iter.Seq2[LLMChunk, error]
}

// LLMRequest is what an LLM is asked.
type LLMRequest struct {
	// System is the system prompt, in blocks: the stable ones first, so that a model can cache them.
	System []string

	// Messages is the conversation, oldest first.
	Messages []LLMMessage

	// Tools are the actions the model may call, with their JSON Schemas.
	Tools []ActionDescriptor
}

// LLMMessage is a message of a conversation with a model.
type LLMMessage struct {
	Role  Role
	Parts []LLMPart
}

// LLMPart is a part of a message: text, a tool call, a tool's result, or the model's thinking (Thinking and its
// Signature, or RedactedThinking), given back as the model gave it.
type LLMPart struct {
	Text             string
	ToolCall         *LLMToolCall
	ToolResult       *LLMToolResult
	Thinking         string
	Signature        string
	RedactedThinking string
}

// LLMToolCall is a model's call of a tool: an action request.
type LLMToolCall struct {
	ID    string
	Name  string
	Input map[string]any
}

// LLMToolResult is the result of a tool call: an action result.
type LLMToolResult struct {
	CallID  string
	Content string
}

// LLMChunk is a piece of what an LLM streams: a piece of text, a whole tool call, or a whole block of thinking
// (Thinking and its Signature, or RedactedThinking), which is to be given back as it came.
type LLMChunk struct {
	Text             string
	ToolCall         *LLMToolCall
	Thinking         string
	Signature        string
	RedactedThinking string
}

// System1Model is a System One model, as Jev (TypeSafe) defines one: given a state and a question about it, it
// answers in one pass, with calibrated probabilities. It generates no text: it decides among options defined in
// advance (a choice), rates on a rubric (a score), or tells how likely a statement is (a noul), so it cannot extract
// free-form values. The state is what the question is about: a short, dense text, or JSON (the conversation, the
// program's state). Ask several questions about the same state concurrently; pair a choice with a noul ("does any
// option apply?") rather than add a "none" option, and gate on the confidence as well as on the choice.
type System1Model interface {
	// AnswerChoice picks the option of the question that best fits the state.
	AnswerChoice(ctx context.Context, state string, q System1Question) (System1ChoiceAnswer, error)
	// AnswerScore rates the state on the question's rubric.
	AnswerScore(ctx context.Context, state string, q System1Question) (System1ScoreAnswer, error)
	// AnswerNoul tells how likely the question's statement is true of the state.
	AnswerNoul(ctx context.Context, state string, q System1Question) (System1NoulAnswer, error)
}

// System1Question asks to pick one option of Criteria.
type System1Question struct {
	Instructions string `json:"instructions"`
	// Criteria are the options, by name, and what each means (up to 255).
	Criteria map[string]string `json:"criteria,omitempty"`
}

// System1ChoiceAnswer is the answer to a System1Question.
type System1ChoiceAnswer struct {
	// Choice is the most probable option.
	Choice string `json:"choice"`
	// Probabilities are those of the options, summing to 1.
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	// Confidence is how sure the model is of the choice, from 0 to 1.
	Confidence float64 `json:"confidence"`
}

// System1ScoreAnswer is the answer to a ScoreQuestion.
type System1ScoreAnswer struct {
	// Score is the rating, normalized from 0 (the lowest level of the rubric) to 1 (its highest).
	Score float64 `json:"score"`
	// Probabilities are those of the levels of the rubric, by name, summing to 1.
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	// Confidence is how sure the model is of the score, from 0 to 1.
	Confidence float64 `json:"confidence"`
}

// System1NoulAnswer is the answer to a NoulQuestion.
type System1NoulAnswer struct {
	// Noul is the probability, from 0 to 1, that the statement is true.
	Noul float64 `json:"noul"`
}
