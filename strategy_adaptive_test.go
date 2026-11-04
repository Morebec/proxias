package proxias_test

import (
	"context"
	"errors"
	"iter"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

// fixedSelector is a StrategySelector that chooses the same name every time,
// or fails, and counts its calls.
type fixedSelector struct {
	name  string
	err   error
	calls *int
}

func (s fixedSelector) Select(context.Context, proxias.Invocation, []proxias.StrategyCandidate) (string, error) {
	*s.calls++
	return s.name, s.err
}

// sayingStrategy is a strategy whose turn is a message of the agent, saying
// its name, or an error when it fails.
func sayingStrategy(name string, err error) proxias.CognitiveStrategy {
	return proxias.StrategyFunc(
		func(_ context.Context, inv proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
			return func(yield func(proxias.StreamUpdate) bool) {
				if err != nil {
					yield(proxias.StreamUpdate{Err: err})
					return
				}
				yield(
					proxias.StreamUpdate{
						Events: []proxias.Event{
							proxias.NewMessageEvent(inv.Agent.AsSource(), name),
						},
						StopReason: proxias.StopReasonEndTurn,
					},
				)
			}
		},
	)
}

func TestAdaptiveStrategy(t *testing.T) {
	tests := []struct {
		name string
		// chooses is what the selector chooses; fails, its error; defaultName
		// the strategy's Default; failing, the candidate whose turn fails.
		chooses     string
		fails       error
		defaultName string
		failing     string
		// given is the session after John's message.
		given []proxias.Event
		// then are the events of the turn: their type and payload; selected
		// how many times the selector was asked.
		then     []string
		selected int
		err      string
	}{
		{
			name:    "GIVEN a person's message THEN the selector chooses, the choice is recorded, and the turn is the chosen strategy's",
			chooses: "rewoo",
			then: []string{
				`cognition.thought {"strategy":"rewoo"}`,
				"message rewoo",
			},
			selected: 1,
		},
		{
			name:    "GIVEN the selector chooses none THEN the task is the first candidate's, recorded as such",
			chooses: "",
			then: []string{
				`cognition.thought {"strategy":"react"}`,
				"message react",
			},
			selected: 1,
		},
		{
			name:        "GIVEN the selector chooses none, and a Default THEN the task is the Default's",
			chooses:     "",
			defaultName: "observe-plan-execute",
			then: []string{
				`cognition.thought {"strategy":"observe-plan-execute"}`,
				"message observe-plan-execute",
			},
			selected: 1,
		},
		{
			name:    "GIVEN the selector chooses no candidate's name THEN the task is the default's",
			chooses: "reflexion",
			then: []string{
				`cognition.thought {"strategy":"react"}`,
				"message react",
			},
			selected: 1,
		},
		{
			name:    "GIVEN a recorded choice, then an action's result THEN the task keeps to its strategy, the selector not asked",
			chooses: "react",
			given: []proxias.Event{
				proxias.NewEvent(proxias.EventTypeThought, proxias.NewAgentSource("ada"), []byte(`{"strategy":"rewoo"}`)),
				proxias.NewEvent(proxias.EventTypeActionRequest, proxias.NewAgentSource("ada"), []byte(`{"action_id":"send_email"}`)),
				proxias.NewEvent(proxias.EventTypeActionResult, proxias.NewSystemSource("actions"), []byte(`{"status":"completed"}`)),
			},
			then: []string{
				"message rewoo",
			},
		},
		{
			name:    "GIVEN a recorded choice, then the agent's plan and a system's message THEN the message is a new task: the selector chooses again",
			chooses: "react",
			given: []proxias.Event{
				proxias.NewEvent(proxias.EventTypeThought, proxias.NewAgentSource("ada"), []byte(`{"strategy":"observe-plan-execute"}`)),
				proxias.NewEvent(proxias.EventTypeThought, proxias.NewAgentSource("ada"), []byte(`{"steps":[{"action":"send_email"}]}`)),
				proxias.NewMessageEvent(proxias.NewSystemSource("clock"), "It is noon."),
			},
			then: []string{
				`cognition.thought {"strategy":"react"}`,
				"message react",
			},
			selected: 1,
		},
		{
			name:    "GIVEN a finished task, then a new message THEN the selector chooses again: a new task",
			chooses: "react",
			given: []proxias.Event{
				proxias.NewEvent(proxias.EventTypeThought, proxias.NewAgentSource("ada"), []byte(`{"strategy":"rewoo"}`)),
				proxias.NewMessageEvent(proxias.NewAgentSource("ada"), "I emailed Globex."),
				proxias.NewMessageEvent(proxias.NewUserSource("john"), "Thanks!"),
			},
			then: []string{
				`cognition.thought {"strategy":"react"}`,
				"message react",
			},
			selected: 1,
		},
		{
			name:     "GIVEN the selector fails THEN the error ends the turn, nothing recorded",
			fails:    errors.New("selecting a strategy: down"),
			selected: 1,
			err:      "selecting a strategy: down",
		},
		{
			name:    "GIVEN the chosen strategy fails THEN its error ends the turn, after the choice",
			chooses: "rewoo",
			failing: "rewoo",
			then: []string{
				`cognition.thought {"strategy":"rewoo"}`,
			},
			selected: 1,
			err:      "rewoo failed",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selected := 0
			candidates := make([]proxias.StrategyCandidate, 0, 3)
			for _, name := range []string{"react", "rewoo", "observe-plan-execute"} {
				var err error
				if name == tt.failing {
					err = errors.New(name + " failed")
				}
				candidate := proxias.StrategyCandidate{
					Name:        name,
					Description: "the " + name + " strategy",
					Strategy:    sayingStrategy(name, err),
				}
				candidates = append(candidates, candidate)
			}
			strategy := proxias.NewAdaptiveStrategy(
				fixedSelector{
					name:  tt.chooses,
					err:   tt.fails,
					calls: &selected,
				},
				candidates...,
			)
			strategy.Default = tt.defaultName
			given := append(
				[]proxias.Event{
					proxias.NewMessageEvent(proxias.NewUserSource("john"), "Email Globex."),
				},
				tt.given...,
			)
			inv := proxias.Invocation{
				Agent:   proxias.Agent{ID: "ada"},
				Session: proxias.NewSession(given...),
			}

			var (
				events []string
				err    error
			)
			for u := range strategy.Invoke(context.Background(), inv) {
				if u.Err != nil {
					err = u.Err
				}
				for _, e := range u.Events {
					events = append(events, string(e.Type)+" "+e.Text())
				}
			}

			if tt.err != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, tt.then, events)
			assert.Equal(t, tt.selected, selected)
		})
	}
}

func TestLLMStrategySelector(t *testing.T) {
	tests := []struct {
		name   string
		answer string
		then   string
		// told is what the model is told, among others; err the error.
		told []string
		err  string
	}{
		{
			name:   "GIVEN the model answers a strategy THEN it is the choice, the model told the candidates and the agent's actions without their schemas",
			answer: `{"strategy":"rewoo"}`,
			then:   "rewoo",
			told: []string{
				`{"name":"rewoo","description":"Plans every step up front."}`,
				`"id":"send_email","name":"","description":"Sends an email."`,
			},
		},
		{
			name:   "GIVEN the model answers in a code fence THEN its JSON is read",
			answer: "```json\n{\"strategy\":\"react\"}\n```",
			then:   "react",
		},
		{
			name:   "GIVEN the model answers none THEN there is no choice",
			answer: `{"strategy":""}`,
			then:   "",
		},
		{
			name:   "GIVEN the model answers no JSON THEN it is an error",
			answer: "ReWOO, I think.",
			err:    "selecting a strategy: the model's answer has no JSON object",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm := &scriptedLLM{answers: []string{tt.answer}}
			agent := proxias.Agent{
				ID: "ada",
				ActionExecutor: proxias.SimpleActionExecutor{
					Actions: map[string]proxias.SimpleAction{
						"send_email": {
							ActionDescriptor: proxias.ActionDescriptor{
								ID:          "send_email",
								Description: "Sends an email.",
								InputSchema: map[string]any{
									"type": "object",
								},
							},
						},
					},
				},
			}
			inv := proxias.Invocation{
				Agent:   agent,
				Session: proxias.NewSession(proxias.NewMessageEvent(proxias.NewUserSource("john"), "Email Globex.")),
			}
			candidates := []proxias.StrategyCandidate{
				{
					Name:        "react",
					Description: "Decides each step after the last one.",
				},
				{
					Name:        "rewoo",
					Description: "Plans every step up front.",
				},
			}

			name, err := proxias.LLMStrategySelector{LLM: llm}.Select(context.Background(), inv, candidates)

			if tt.err != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.then, name)
			require.Len(t, llm.asked, 1)
			told := systemText(llm.asked[0])
			for _, s := range tt.told {
				assert.Contains(t, told, s)
			}
			assert.NotContains(t, told, `"schema"`, "the actions are told without their schemas")
		})
	}
}

// choosingSystem1 is a System1Model whose choice is fixed, and which records
// the question it is asked.
type choosingSystem1 struct {
	choice proxias.System1ChoiceAnswer
	asked  *proxias.System1Question
}

func (m choosingSystem1) AnswerChoice(_ context.Context, _ string, q proxias.System1Question) (proxias.System1ChoiceAnswer, error) {
	*m.asked = q
	return m.choice, nil
}

func (m choosingSystem1) AnswerScore(context.Context, string, proxias.System1Question) (proxias.System1ScoreAnswer, error) {
	return proxias.System1ScoreAnswer{}, errors.New("not asked")
}

func (m choosingSystem1) AnswerNoul(context.Context, string, proxias.System1Question) (proxias.System1NoulAnswer, error) {
	return proxias.System1NoulAnswer{}, errors.New("not asked")
}

func TestSystem1StrategySelector(t *testing.T) {
	zero := 0.0
	tests := []struct {
		name      string
		choice    proxias.System1ChoiceAnswer
		threshold *float64
		then      string
	}{
		{
			name: "GIVEN a confident choice THEN it is the choice",
			choice: proxias.System1ChoiceAnswer{
				Choice:     "rewoo",
				Confidence: 0.9,
			},
			then: "rewoo",
		},
		{
			name: "GIVEN a choice under the default threshold THEN there is no choice",
			choice: proxias.System1ChoiceAnswer{
				Choice:     "rewoo",
				Confidence: 0.3,
			},
			then: "",
		},
		{
			name: "GIVEN a threshold of 0 THEN whatever System1 chooses is the choice",
			choice: proxias.System1ChoiceAnswer{
				Choice:     "react",
				Confidence: 0.1,
			},
			threshold: &zero,
			then:      "react",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var asked proxias.System1Question
			selector := proxias.System1StrategySelector{
				System1: choosingSystem1{
					choice: tt.choice,
					asked:  &asked,
				},
				SelectionThreshold: tt.threshold,
			}
			inv := proxias.Invocation{
				Agent:   proxias.Agent{ID: "ada"},
				Session: proxias.NewSession(proxias.NewMessageEvent(proxias.NewUserSource("john"), "Email Globex.")),
			}
			candidates := []proxias.StrategyCandidate{
				{
					Name:        "react",
					Description: "Decides each step after the last one.",
				},
				{
					Name:        "rewoo",
					Description: "Plans every step up front.",
				},
			}

			name, err := selector.Select(context.Background(), inv, candidates)

			require.NoError(t, err)
			assert.Equal(t, tt.then, name)
			assert.Equal(
				t,
				map[string]string{
					"react": "Decides each step after the last one.",
					"rewoo": "Plans every step up front.",
				},
				asked.Criteria,
				"System1 chooses among the candidates, by their descriptions",
			)
			assert.Equal(t, proxias.DefaultSystem1StrategyChoiceInstructions, asked.Instructions)
		})
	}
}
