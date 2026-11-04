package agentsdk_test

import (
	"context"
	"errors"
	"iter"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
)

// memory is a MemoryProvider of functions, without actions.
type memory struct {
	recall   func(agentsdk.Invocation) ([]agentsdk.Event, error)
	memorize func(agentsdk.Invocation, []agentsdk.Event) error
}

func (m memory) ActionExecutor() agentsdk.ActionExecutor { return nil }

func (m memory) Recall(_ context.Context, inv agentsdk.Invocation) ([]agentsdk.Event, error) {
	return m.recall(inv)
}

func (m memory) Memorize(_ context.Context, inv agentsdk.Invocation, events []agentsdk.Event) error {
	return m.memorize(inv, events)
}

func TestAgentMemory(t *testing.T) {
	tests := []struct {
		name string
		// memory is the agent's: none when it recalls nothing and memorizes
		// nothing.
		memory   bool
		recalled []agentsdk.Event
		recall   error
		memorize error
		// told are the memories the agent's prompt tells, memorized the events the
		// memory is given, then the events of the
		// stream, err its error.
		told, memorized, then []string
		err                   string
	}{
		{
			name:   "GIVEN a memory THEN the turn is given what it recalls, not in its session, and it memorizes what the turn added",
			memory: true,
			recalled: []agentsdk.Event{
				agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "the demo is on friday"),
			},
			told:      []string{"the demo is on friday"},
			memorized: []string{"Friday."},
			then:      []string{"when is the demo?", "Friday."},
		},
		{
			name: "GIVEN no memory THEN the turn is given no memories",
			then: []string{"when is the demo?", "Friday."},
		},
		{
			name:   "GIVEN the memory fails to recall THEN the error ends the turn, before the strategy",
			memory: true,
			recall: errors.New("disk"),
			then:   []string{"when is the demo?"},
			err:    "recalling: disk",
		},
		{
			name:      "GIVEN the memory fails to memorize THEN the error ends the turn, after its events",
			memory:    true,
			memorize:  errors.New("disk"),
			memorized: []string{"Friday."},
			then:      []string{"when is the demo?", "Friday."},
			err:       "memorizing: disk",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var told string
			var memorized []string
			agent := agentsdk.Agent{ID: "ada"}
			if tt.memory {
				agent.Memory = memory{
					recall: func(agentsdk.Invocation) ([]agentsdk.Event, error) { return tt.recalled, tt.recall },
					memorize: func(_ agentsdk.Invocation, events []agentsdk.Event) error {
						for _, e := range events {
							memorized = append(memorized, e.Text())
						}
						return tt.memorize
					},
				}
			}
			strategy := agentsdk.StrategyFunc(
				func(ctx context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
					return func(yield func(agentsdk.StreamUpdate) bool) {
						agentPrompt, err := agentsdk.NewAgentPrompt(ctx, inv)
						if err != nil {
							yield(agentsdk.StreamUpdate{Err: err})
							return
						}
						told = agentPrompt.String()
						yield(
							agentsdk.StreamUpdate{
								Events: []agentsdk.Event{
									agentsdk.NewMessageEvent(agentsdk.NewAgentSource("ada"), "Friday."),
								},
								StopReason: agentsdk.StopReasonEndTurn,
							},
						)
					}
				},
			)

			stream, err := agent.Invoke(
				context.Background(),
				agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "when is the demo?")),
				strategy,
			)
			require.NoError(t, err)
			session, err := stream.Collect()

			if tt.err != "" {
				require.EqualError(t, err, tt.err)
			} else {
				require.NoError(t, err)
			}
			then := make([]string, 0, len(session.Events()))
			for _, e := range session.Events() {
				then = append(then, e.Text())
			}
			assert.Equal(t, tt.then, then)
			for _, m := range tt.told {
				assert.Contains(t, told, m)
			}
			assert.Equal(t, tt.memorized, memorized)
		})
	}
}

func TestMarkdownMemory(t *testing.T) {
	tests := []struct {
		name string
		// given is the document, absent when nil; update the content
		// update_memory is given, when set.
		given  *string
		update string
		// recalled are the memories recalled after the update, then the
		// document.
		recalled []string
		then     string
	}{
		{
			name:     "GIVEN no document THEN nothing is recalled",
			recalled: []string{},
		},
		{
			name:     "GIVEN an empty document THEN nothing is recalled",
			given:    new(string),
			recalled: []string{},
		},
		{
			name:     "GIVEN a document THEN it is recalled whole, as one memory",
			given:    func() *string { s := "# John\n- likes tea"; return &s }(),
			recalled: []string{"# John\n- likes tea"},
			then:     "# John\n- likes tea",
		},
		{
			name:     "GIVEN no document, and an update THEN the update creates it, and it is recalled",
			update:   "# John\n- likes tea",
			recalled: []string{"# John\n- likes tea"},
			then:     "# John\n- likes tea",
		},
		{
			name:     "GIVEN a document, and an update THEN the update replaces it whole",
			given:    func() *string { s := "# John\n- likes tea\n- lives in Paris"; return &s }(),
			update:   "# John\n- likes coffee",
			recalled: []string{"# John\n- likes coffee"},
			then:     "# John\n- likes coffee",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "memory", "MEMORY.md")
			if tt.given != nil {
				require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o750))
				require.NoError(t, os.WriteFile(path, []byte(*tt.given), 0o600))
			}
			memory := agentsdk.MarkdownMemory{Path: path}
			inv := agentsdk.Invocation{
				Session: agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hi")),
			}

			descriptors, err := memory.ActionExecutor().
				Descriptors(context.Background(), inv.Session)
			require.NoError(t, err)
			require.Len(t, descriptors, 1)
			assert.Equal(t, "update_memory", descriptors[0].ID)
			assert.Equal(t, agentsdk.DefaultMarkdownMemoryUpdateDescription, descriptors[0].Description)
			if tt.update != "" {
				results, err := memory.ActionExecutor().
					Execute(context.Background(), inv.Session, []agentsdk.ActionExecutionRequest{{ActionID: "update_memory", Params: map[string]any{"content": tt.update}}})
				require.NoError(t, err)
				assert.Equal(
					t,
					[]agentsdk.ActionExecutionResult{
						{ActionID: "update_memory", Status: "completed"},
					},
					results,
				)
			}
			require.NoError(
				t,
				memory.Memorize(
					context.Background(),
					inv,
					[]agentsdk.Event{
						agentsdk.NewMessageEvent(agentsdk.NewAgentSource("ada"), "ignored"),
					},
				),
			)
			memories, err := memory.Recall(context.Background(), inv)

			require.NoError(t, err)
			recalled := make([]string, 0, len(memories))
			for _, m := range memories {
				recalled = append(recalled, m.Text())
			}
			assert.Equal(t, tt.recalled, recalled)
			document, _ := os.ReadFile(path)
			assert.Equal(t, tt.then, string(document))
		})
	}
}

// personal is a memory of a document per person and a shared one, by person ID
// ("" for the shared one): the interface lets it tell, at Recall and when its
// action runs, whom the agent deals with, from the session.
type personal map[string]string

// person is the source of the session's last message from a person.
func person(s agentsdk.Session) string {
	events := s.Events()
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].HasSourceType(agentsdk.EventSourceTypeUser) && events[i].Type == agentsdk.EventTypeMessage {
			return events[i].Source.ID
		}
	}
	return ""
}

func (p personal) Recall(_ context.Context, inv agentsdk.Invocation) ([]agentsdk.Event, error) {
	var memories []agentsdk.Event
	for _, id := range []string{"", person(inv.Session)} {
		if p[id] != "" {
			memories = append(memories, agentsdk.NewMessageEvent(agentsdk.NewSystemSource("memory"), p[id]))
		}
	}
	return memories, nil
}

func (p personal) Memorize(context.Context, agentsdk.Invocation, []agentsdk.Event) error {
	return nil
}

func (p personal) ActionExecutor() agentsdk.ActionExecutor {
	return agentsdk.SimpleActionExecutor{
		Actions: map[string]agentsdk.SimpleAction{"update_memory": {
			ActionDescriptor: agentsdk.ActionDescriptor{ID: "update_memory"},
			RunFn: func(_ context.Context, s agentsdk.Session, params map[string]any) (agentsdk.ActionExecutionResult, error) {
				p[person(s)] = params["content"].(string)
				return agentsdk.ActionExecutionResult{
					ActionID: "update_memory",
					Status:   "completed",
				}, nil
			},
		}},
	}
}

func TestAgentMemoryActions(t *testing.T) {
	tests := []struct {
		name string
		// given are the documents of the memory, by person; from the person the
		// agent answers; actions whether the agent
		// has actions of its own.
		given   map[string]string
		from    string
		actions bool
		// told are the memories the agent's prompt tells, offered the actions it is
		// offered, results the statuses of its requests (update_memory, then
		// echo), then the documents after the turn.
		told, offered, results []string
		then                   map[string]string
	}{
		{
			name: "GIVEN a person's memory and a shared one THEN the turn is told both, offered the memory's action next to the agent's, and the update is the person's",
			given: map[string]string{
				"":     "the office is in Montreal",
				"john": "john likes tea",
				"mary": "mary likes coffee",
			},
			from:    "john",
			actions: true,
			told:    []string{"the office is in Montreal", "john likes tea"},
			offered: []string{"echo", "update_memory"},
			results: []string{"completed", "completed"},
			then: map[string]string{
				"":     "the office is in Montreal",
				"john": "john likes green tea",
				"mary": "mary likes coffee",
			},
		},
		{
			name:    "GIVEN an agent without actions THEN the turn is offered the memory's, and an unknown action fails",
			given:   map[string]string{},
			from:    "mary",
			offered: []string{"update_memory"},
			results: []string{"completed", "failed"},
			then:    map[string]string{"mary": "john likes green tea"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			memory := personal(tt.given)
			agent := agentsdk.Agent{ID: "ada", Memory: memory}
			if tt.actions {
				agent.ActionExecutor = agentsdk.SimpleActionExecutor{
					Actions: map[string]agentsdk.SimpleAction{
						"echo": {
							ActionDescriptor: agentsdk.ActionDescriptor{
								ID: "echo",
							},
							RunFn: func(context.Context, agentsdk.Session, map[string]any) (agentsdk.ActionExecutionResult, error) {
								return agentsdk.ActionExecutionResult{
									ActionID: "echo",
									Status:   "completed",
								}, nil
							},
						},
					},
				}
			}
			var told string
			var offered, results []string
			strategy := agentsdk.StrategyFunc(
				func(ctx context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
					return func(yield func(agentsdk.StreamUpdate) bool) {
						agentPrompt, err := agentsdk.NewAgentPrompt(ctx, inv)
						require.NoError(t, err)
						told = agentPrompt.String()
						descriptors, err := inv.Agent.ActionExecutor.Descriptors(ctx, inv.Session)
						require.NoError(t, err)
						for _, d := range descriptors {
							offered = append(offered, d.ID)
						}
						executed, err := inv.Agent.ActionExecutor.Execute(
							ctx,
							inv.Session,
							[]agentsdk.ActionExecutionRequest{
								{
									ActionID: "update_memory",
									Params: map[string]any{
										"content": "john likes green tea",
									},
								},
								{ActionID: "echo"},
							},
						)
						require.NoError(t, err)
						for _, r := range executed {
							results = append(results, r.Status)
						}
						yield(
							agentsdk.StreamUpdate{
								StopReason: agentsdk.StopReasonEndTurn,
							},
						)
					}
				},
			)

			stream, err := agent.Invoke(
				context.Background(),
				agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource(tt.from), "I prefer green tea")),
				strategy,
			)
			require.NoError(t, err)
			_, err = stream.Collect()

			require.NoError(t, err)
			for _, m := range tt.told {
				assert.Contains(t, told, m)
			}
			assert.Equal(t, tt.offered, offered)
			assert.Equal(t, tt.results, results)
			assert.Equal(t, tt.then, map[string]string(memory))
		})
	}
}
