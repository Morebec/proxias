package proxias_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func TestObservePlanExecuteStrategy(t *testing.T) {
	tests := []struct {
		name    string
		answers []string
		// actions are the agent's, by ID, each with the result it gives; none
		// for an agent without actions.
		actions map[string]proxias.ActionExecutionResult
		// given is the session after John's message.
		given []proxias.Event
		// then are the events of the turn: their type, source and payload; ran
		// the params of each execution, by action; told what the last call is
		// told of the steps.
		then  []string
		ran   map[string]map[string]any
		calls int
		told  []string
		stop  proxias.StopReason
		err   string
	}{
		{
			name: "GIVEN a message THEN the turn plans, and yields the plan for the next turn to run",
			answers: []string{
				`{"steps":[{"action":"find_customer","input":{"name":"Globex"}},{"action":"send_email","input":{"to":"?"}}]}`,
			},
			actions: map[string]proxias.ActionExecutionResult{
				"find_customer": {
					ActionID: "find_customer",
					Status:   "completed",
				},
			},
			then: []string{
				`cognition.thought agent:ada {"steps":[{"action":"find_customer","input":{"name":"Globex"}},{"action":"send_email","input":{"to":"?"}}]}`,
			},
			ran:   map[string]map[string]any{},
			calls: 1,
			told: []string{
				"No step has run yet.",
			},
		},
		{
			name: "GIVEN the agent's plan is the last event THEN the turn runs its first step, asking no model",
			actions: map[string]proxias.ActionExecutionResult{
				"find_customer": {
					ActionID: "find_customer",
					Status:   "completed",
					Data: map[string]any{
						"email": "joe@globex.com",
					},
				},
			},
			given: []proxias.Event{
				proxias.NewEvent(
					proxias.EventTypeThought,
					proxias.NewAgentSource("ada"),
					[]byte(`{"steps":[{"action":"find_customer","input":{"name":"Globex"}},{"action":"send_email"}]}`),
				),
			},
			then: []string{
				`action.request agent:ada {"action_id":"find_customer","params":{"name":"Globex"}}`,
				`action.result system:actions {"action_id":"find_customer","status":"completed","data":{"email":"joe@globex.com"}}`,
			},
			ran: map[string]map[string]any{
				"find_customer": {
					"name": "Globex",
				},
			},
		},
		{
			name: "GIVEN the agent's plan, then a system's instruction (a clock) THEN the turn still runs the plan's first step",
			actions: map[string]proxias.ActionExecutionResult{
				"send_email": {
					ActionID: "send_email",
					Status:   "completed",
				},
			},
			given: []proxias.Event{
				proxias.NewEvent(
					proxias.EventTypeThought,
					proxias.NewAgentSource("ada"),
					[]byte(`{"steps":[{"action":"send_email","input":{"to":"joe@globex.com"}}]}`),
				),
				proxias.NewMessageEvent(proxias.NewSystemSource("clock"), "It is noon."),
			},
			then: []string{
				`action.request agent:ada {"action_id":"send_email","params":{"to":"joe@globex.com"}}`,
				`action.result system:actions {"action_id":"send_email","status":"completed"}`,
			},
			ran: map[string]map[string]any{
				"send_email": {
					"to": "joe@globex.com",
				},
			},
		},
		{
			name: "GIVEN a step's result, then a system's instruction THEN the plan is still told what the step found",
			answers: []string{
				`{"steps":[]}`,
				"I found Globex.",
			},
			actions: map[string]proxias.ActionExecutionResult{
				"find_customer": {
					ActionID: "find_customer",
					Status:   "completed",
				},
			},
			given: []proxias.Event{
				proxias.NewEvent(
					proxias.EventTypeActionRequest,
					proxias.NewAgentSource("ada"),
					[]byte(`{"action_id":"find_customer","params":{"name":"Globex"}}`),
				),
				proxias.NewEvent(
					proxias.EventTypeActionResult,
					proxias.NewSystemSource("actions"),
					[]byte(`{"action_id":"find_customer","status":"completed"}`),
				),
				proxias.NewMessageEvent(proxias.NewSystemSource("clock"), "It is noon."),
			},
			then: []string{
				"message agent:ada I found Globex.",
			},
			ran:   map[string]map[string]any{},
			calls: 2,
			told: []string{
				`- {"action_id":"find_customer","params":{"name":"Globex"}}: {"action_id":"find_customer","status":"completed"}`,
			},
			stop: proxias.StopReasonEndTurn,
		},
		{
			name: "GIVEN a step's result THEN the turn plans again, told what the step found",
			answers: []string{
				`{"steps":[{"action":"send_email","input":{"to":"joe@globex.com"}}]}`,
			},
			actions: map[string]proxias.ActionExecutionResult{
				"send_email": {
					ActionID: "send_email",
					Status:   "completed",
				},
			},
			given: []proxias.Event{
				proxias.NewEvent(
					proxias.EventTypeActionRequest,
					proxias.NewAgentSource("ada"),
					[]byte(`{"action_id":"find_customer","params":{"name":"Globex"}}`),
				),
				proxias.NewEvent(
					proxias.EventTypeActionResult,
					proxias.NewSystemSource("actions"),
					[]byte(`{"action_id":"find_customer","status":"completed","data":{"email":"joe@globex.com"}}`),
				),
			},
			then: []string{
				`cognition.thought agent:ada {"steps":[{"action":"send_email","input":{"to":"joe@globex.com"}}]}`,
			},
			ran:   map[string]map[string]any{},
			calls: 1,
			told: []string{
				`- {"action_id":"find_customer","params":{"name":"Globex"}}: {"action_id":"find_customer","status":"completed","data":{"email":"joe@globex.com"}}`,
			},
		},
		{
			name: "GIVEN a plan of no step left THEN the turn reports, told what the steps found, with the model's stop reason",
			answers: []string{
				`{"steps":[]}`,
				"I emailed Globex.",
			},
			actions: map[string]proxias.ActionExecutionResult{
				"send_email": {
					ActionID: "send_email",
					Status:   "completed",
				},
			},
			given: []proxias.Event{
				proxias.NewEvent(
					proxias.EventTypeActionRequest,
					proxias.NewAgentSource("ada"),
					[]byte(`{"action_id":"send_email","params":{"to":"joe@globex.com"}}`),
				),
				proxias.NewEvent(
					proxias.EventTypeActionResult,
					proxias.NewSystemSource("actions"),
					[]byte(`{"action_id":"send_email","status":"completed"}`),
				),
			},
			then: []string{
				"message agent:ada I emailed Globex.",
			},
			ran:   map[string]map[string]any{},
			calls: 2,
			told: []string{
				`- {"action_id":"send_email","params":{"to":"joe@globex.com"}}: {"action_id":"send_email","status":"completed"}`,
			},
			stop: proxias.StopReasonEndTurn,
		},
		{
			name: "GIVEN an agent without actions THEN the turn reports, asking no plan",
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
			name: "GIVEN a plan that is not JSON THEN it is an error",
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
			given := append(
				[]proxias.Event{
					proxias.NewMessageEvent(proxias.NewUserSource("john"), "Email Globex."),
				},
				tt.given...,
			)
			inv := proxias.Invocation{
				Agent:   agent,
				Session: proxias.NewSession(given...),
			}

			var (
				events []string
				stop   proxias.StopReason
				err    error
			)
			for u := range proxias.NewObservePlanExecuteStrategy(llm).Invoke(context.Background(), inv) {
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

func TestObservePlanExecuteStrategyLoop(t *testing.T) {
	llm := &scriptedLLM{
		answers: []string{
			`{"steps":[{"action":"find_customer","input":{"name":"Globex"}},{"action":"send_email"}]}`,
			`{"steps":[{"action":"send_email","input":{"to":"joe@globex.com"}}]}`,
			`{"steps":[]}`,
			"I emailed joe@globex.com.",
		},
	}
	agent := proxias.Agent{
		ID: "ada",
		ActionExecutor: proxias.SimpleActionExecutor{
			Actions: map[string]proxias.SimpleAction{
				"find_customer": {
					ActionDescriptor: proxias.ActionDescriptor{
						ID: "find_customer",
					},
					RunFn: func(context.Context, proxias.Session, map[string]any) (proxias.ActionExecutionResult, error) {
						return proxias.ActionExecutionResult{
							ActionID: "find_customer",
							Status:   "completed",
							Data: map[string]any{
								"email": "joe@globex.com",
							},
						}, nil
					},
				},
				"send_email": {
					ActionDescriptor: proxias.ActionDescriptor{
						ID: "send_email",
					},
					RunFn: func(context.Context, proxias.Session, map[string]any) (proxias.ActionExecutionResult, error) {
						return proxias.ActionExecutionResult{
							ActionID: "send_email",
							Status:   "completed",
						}, nil
					},
				},
			},
		},
	}

	stream, err := agent.Loop(
		context.Background(),
		proxias.NewSession(proxias.NewMessageEvent(proxias.NewUserSource("john"), "Email Globex.")),
		proxias.NewObservePlanExecuteStrategy(llm),
	)
	require.NoError(t, err)
	session, err := stream.Collect()
	require.NoError(t, err)

	events := make([]string, 0, len(session.Events()))
	for _, e := range session.Events() {
		events = append(events, string(e.Type)+" "+e.Text())
	}
	assert.Equal(
		t,
		[]string{
			"message Email Globex.",
			`cognition.thought {"steps":[{"action":"find_customer","input":{"name":"Globex"}},{"action":"send_email"}]}`,
			`action.request {"action_id":"find_customer","params":{"name":"Globex"}}`,
			`action.result {"action_id":"find_customer","status":"completed","data":{"email":"joe@globex.com"}}`,
			`cognition.thought {"steps":[{"action":"send_email","input":{"to":"joe@globex.com"}}]}`,
			`action.request {"action_id":"send_email","params":{"to":"joe@globex.com"}}`,
			`action.result {"action_id":"send_email","status":"completed"}`,
			"message I emailed joe@globex.com.",
		},
		events,
		"each step is planned, then run; the plan is revised from what it found",
	)
	assert.Len(t, llm.asked, 4, "a plan before each step, the last one empty, then the answer")
	assert.Equal(t, proxias.StopReasonEndTurn, stream.StopReason())
}
