package proxias

import "context"

// System1Model is a System One model, as Jev (TypeSafe) defines one: given a
// state and a question about it, it answers in one pass, with calibrated
// probabilities. It generates no text: it decides among options defined in
// advance (a choice), rates on a rubric (a score), or tells how likely a
// statement is (a noul), so it cannot extract free-form values. The state is
// what the question is about: a short, dense text, or JSON (the conversation,
// the program's state). Ask several questions about the same state
// concurrently; pair a choice with a noul ("does any option apply?") rather
// than add a "none" option, and gate on the confidence as well as on the
// choice.
type System1Model interface {
	// AnswerChoice picks the option of the question that best fits the state.
	AnswerChoice(ctx context.Context, state string, q System1Question) (System1ChoiceAnswer, error)
	// AnswerScore rates the state on the question's rubric.
	AnswerScore(ctx context.Context, state string, q System1Question) (System1ScoreAnswer, error)
	// AnswerNoul tells how likely the question's statement is true of the
	// state.
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
	// Score is the rating, normalized from 0 (the lowest level of the rubric)
	// to 1 (its highest).
	Score float64 `json:"score"`
	// Probabilities are those of the levels of the rubric, by name, summing to
	// 1.
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	// Confidence is how sure the model is of the score, from 0 to 1.
	Confidence float64 `json:"confidence"`
}

// System1NoulAnswer is the answer to a NoulQuestion.
type System1NoulAnswer struct {
	// Noul is the probability, from 0 to 1, that the statement is true.
	Noul float64 `json:"noul"`
}
