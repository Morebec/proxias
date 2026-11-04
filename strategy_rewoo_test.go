package proxias_test

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

// scriptedLLM is an LLM that answers its texts in order, one a call, each
// ended by end_turn, and records what it is asked.
type scriptedLLM struct {
	answers []string
	asked   []proxias.LLMRequest
}

func (l *scriptedLLM) Complete(_ context.Context, req proxias.LLMRequest) iter.Seq2[proxias.LLMChunk, error] {
	l.asked = append(l.asked, req)
	return func(yield func(proxias.LLMChunk, error) bool) {
		call := len(l.asked) - 1
		if call >= len(l.answers) {
			yield(proxias.LLMChunk{}, errors.New("no answer scripted"))
			return
		}
		if !yield(proxias.LLMChunk{Text: l.answers[call]}, nil) {
			return
		}
		yield(proxias.LLMChunk{StopReason: proxias.StopReasonEndTurn}, nil)
	}
}

// systemText is the text of the system messages of a request.
func systemText(req proxias.LLMRequest) string {
	var text []string
	for _, message := range req.Messages {
		if message.Role != proxias.LLMMessageRoleSystem {
			continue
		}
		for _, part := range message.Parts {
			text = append(text, part.Text)
		}
	}
	return strings.Join(text, "\n")
}

func TestReWOOStrategy(t *testing.T) {
	tests := []struct {
		name    string
		answers []string
		// actions are the agent's, by ID, each with the result it gives; none
		// for an agent without actions.
		actions map[string]proxias.ActionExecutionResult
		// then are the events of the turn: their type, source and payload; ran
		// the params of each execution, by action; told what the answer is told
		// of the plan.
		then  []string
		ran   map[string]map[string]any
		calls int
		told  []string
		stop  proxias.StopReason
		err   string
	}{
		{
			name: "GIVEN a plan whose second step names the first's result THEN they run in order, the reference replaced, and the answer is written from what they found",
			answers: []string{
				`{"steps":[{"id":"E1","action":"find_customer","input":{"name":"Globex"}},{"id":"E2","action":"send_email","input":{"to":"#E1.email","body":"Hello"}}]}`,
				"I emailed Globex.",
			},
			actions: map[string]proxias.ActionExecutionResult{
				"find_customer": {
					ActionID: "find_customer",
					Status:   "completed",
					Data: map[string]any{
						"email": "joe@globex.com",
					},
				},
				"send_email": {
					ActionID: "send_email",
					Status:   "completed",
				},
			},
			then: []string{
				`cognition.thought agent:ada {"steps":[{"id":"E1","action":"find_customer","input":{"name":"Globex"}},{"id":"E2","action":"send_email","input":{"body":"Hello","to":"#E1.email"}}]}`,
				`action.request agent:ada {"action_id":"find_customer","params":{"name":"Globex"}}`,
				`action.result system:actions {"action_id":"find_customer","status":"completed","data":{"email":"joe@globex.com"}}`,
				`action.request agent:ada {"action_id":"send_email","params":{"body":"Hello","to":"joe@globex.com"}}`,
				`action.result system:actions {"action_id":"send_email","status":"completed"}`,
				"message agent:ada I emailed Globex.",
			},
			ran: map[string]map[string]any{
				"find_customer": {
					"name": "Globex",
				},
				"send_email": {
					"body": "Hello",
					"to":   "joe@globex.com",
				},
			},
			calls: 2,
			told: []string{
				`#E1 = find_customer({"name":"Globex"}): completed {"email":"joe@globex.com"}`,
				`#E2 = send_email({"body":"Hello","to":"#E1.email"}): completed null`,
			},
			stop: proxias.StopReasonEndTurn,
		},
		{
			name: "GIVEN a step that fails THEN the work stops, and the answer is told the next step was not run",
			answers: []string{
				`{"steps":[{"id":"E1","action":"find_customer","input":{"name":"Initech"}},{"id":"E2","action":"send_email","input":{"to":"#E1.email"}}]}`,
				"I could not find Initech.",
			},
			actions: map[string]proxias.ActionExecutionResult{
				"find_customer": {
					ActionID: "find_customer",
					Status:   "failed",
					Data: map[string]any{
						"error": "not found",
					},
				},
				"send_email": {
					ActionID: "send_email",
					Status:   "completed",
				},
			},
			then: []string{
				`cognition.thought agent:ada {"steps":[{"id":"E1","action":"find_customer","input":{"name":"Initech"}},{"id":"E2","action":"send_email","input":{"to":"#E1.email"}}]}`,
				`action.request agent:ada {"action_id":"find_customer","params":{"name":"Initech"}}`,
				`action.result system:actions {"action_id":"find_customer","status":"failed","data":{"error":"not found"}}`,
				"message agent:ada I could not find Initech.",
			},
			ran: map[string]map[string]any{
				"find_customer": {
					"name": "Initech",
				},
			},
			calls: 2,
			told: []string{
				`#E1 = find_customer({"name":"Initech"}): failed {"error":"not found"}`,
				`#E2 = send_email({"to":"#E1.email"}): not run`,
			},
			stop: proxias.StopReasonEndTurn,
		},
		{
			name: "GIVEN a plan of no step THEN the answer is written from the conversation, no action run",
			answers: []string{
				`{"steps":[]}`,
				"Hello John.",
			},
			actions: map[string]proxias.ActionExecutionResult{
				"send_email": {
					ActionID: "send_email",
					Status:   "completed",
				},
			},
			then: []string{
				"message agent:ada Hello John.",
			},
			ran:   map[string]map[string]any{},
			calls: 2,
			stop:  proxias.StopReasonEndTurn,
		},
		{
			name: "GIVEN an agent without actions THEN no plan is asked for: one call, the answer",
			answers: []string{
				"Hello John.",
			},
			then: []string{
				"message agent:ada Hello John.",
			},
			ran:   map[string]map[string]any{},
			calls: 1,
			stop:  proxias.StopReasonEndTurn,
		},
		{
			name: "GIVEN a plan that is not JSON THEN it is an error, and nothing runs",
			answers: []string{
				"I would email Globex.",
			},
			actions: map[string]proxias.ActionExecutionResult{
				"send_email": {
					ActionID: "send_email",
					Status:   "completed",
				},
			},
			ran:   map[string]map[string]any{},
			calls: 1,
			err:   "planning: the model's answer has no JSON object",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm := &scriptedLLM{answers: tt.answers}
			ran := map[string]map[string]any{}
			agent := proxias.Agent{ID: "ada"}
			if tt.actions != nil {
				actions := map[string]proxias.SimpleAction{}
				for id, result := range tt.actions {
					actions[id] = proxias.SimpleAction{
						ActionDescriptor: proxias.ActionDescriptor{
							ID: id,
						},
						RunFn: func(_ context.Context, _ proxias.Session, params map[string]any) (proxias.ActionExecutionResult, error) {
							ran[id] = params
							return result, nil
						},
					}
				}
				agent.ActionExecutor = proxias.SimpleActionExecutor{
					Actions: actions,
				}
			}
			inv := proxias.Invocation{
				Agent:   agent,
				Session: proxias.NewSession(proxias.NewMessageEvent(proxias.NewUserSource("john"), "Email Globex.")),
			}

			var (
				events []string
				stop   proxias.StopReason
				err    error
			)
			for u := range proxias.NewReWOOStrategy(llm).Invoke(context.Background(), inv) {
				if u.Err != nil {
					err = u.Err
				}
				if u.StopReason != "" {
					stop = u.StopReason
				}
				for _, e := range u.Events {
					events = append(events, string(e.Type)+" "+e.Source.String()+" "+e.Text())
				}
			}

			if tt.err != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, tt.then, events)
			assert.Equal(t, tt.ran, ran)
			assert.Len(t, llm.asked, tt.calls)
			assert.Equal(t, tt.stop, stop)
			for _, told := range tt.told {
				assert.Contains(t, systemText(llm.asked[len(llm.asked)-1]), told)
			}
		})
	}
}
