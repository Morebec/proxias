package proxias_test

import (
	"context"
	"errors"
	"iter"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func TestAgentInvoke(t *testing.T) {
	tests := []struct {
		name  string
		given []proxias.StreamUpdate
		err   error
		then  []string
	}{
		{
			name: "GIVEN a turn that streams its answer THEN the chunk comes, then the event",
			given: []proxias.StreamUpdate{
				{
					Chunk: &proxias.StreamChunk{
						EventID: "r",
						Data:    []byte("Hi"),
					},
				},
				{
					Events: []proxias.Event{
						{
							ID:      "r",
							Type:    proxias.EventTypeMessage,
							Source:  proxias.NewAgentSource("ada"),
							Payload: []byte("Hi"),
							Date:    time.Now(),
						},
					},
				},
			},
			then: []string{"chunk Hi", "event Hi"},
		},
		{
			name: "GIVEN a turn that requests an action THEN it is left to the caller",
			given: []proxias.StreamUpdate{
				{
					Events: []proxias.Event{
						proxias.NewEvent(
							proxias.EventTypeActionRequest,
							proxias.NewAgentSource("ada"),
							[]byte(`{"action_id":"a"}`),
						),
					},
				},
			},
			then: []string{`event {"action_id":"a"}`},
		},
		{
			name: "GIVEN a turn that adds nothing THEN there is no update, the stream's session the one given",
		},
		{
			name: "GIVEN a turn that fails THEN the stream ends with the error",
			err:  errors.New("down"),
			then: []string{"error down"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := proxias.StrategyFunc(
				func(context.Context, proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
					return func(yield func(proxias.StreamUpdate) bool) {
						for _, d := range tt.given {
							if !yield(d) {
								return
							}
						}
						if tt.err != nil {
							yield(proxias.StreamUpdate{Err: tt.err})
						}
					}
				},
			)
			agent := proxias.Agent{ID: "ada"}
			start := proxias.NewSession(proxias.NewMessageEvent(proxias.NewUserSource("john"), "hello"))

			var got []string
			stream, err := agent.Invoke(context.Background(), start, strategy)
			require.NoError(t, err)
			for d := range stream.Updates() {
				switch {
				case d.Err != nil:
					got = append(got, "error "+d.Err.Error())
				case d.Chunk != nil:
					got = append(got, "chunk "+string(d.Chunk.Data))
				case len(d.Events) > 0:
					got = append(got, "event "+d.Events[0].Text())
				}
			}
			last := stream

			assert.Equal(t, tt.then, got)
			assert.Equal(t, start.ID(), last.Session().ID())
			for _, e := range last.Session().Events()[1:] {
				assert.NotEmpty(t, e.ID, "the events are stamped")
				assert.Equal(t, proxias.NewAgentSource("ada"), e.Source)
				assert.Equal(t, start.Events()[0].ID, e.ParentID)
			}
		})
	}
}

func TestAgentLoop(t *testing.T) {
	tests := []struct {
		name string
		// given are the updates of each turn.
		given [][]proxias.StreamUpdate
		then  []string
		stop  proxias.StopReason
	}{
		{
			name: "GIVEN a turn that ends its turn THEN the loop ends",
			given: [][]proxias.StreamUpdate{
				{
					{
						Events: []proxias.Event{
							{
								ID:      "1",
								Type:    proxias.EventTypeMessage,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("Hi"),
							},
						},
						StopReason: proxias.StopReasonEndTurn,
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "2",
								Type:    proxias.EventTypeMessage,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("never"),
							},
						},
						StopReason: proxias.StopReasonEndTurn,
					},
				},
			},
			then: []string{"message hello", "message Hi"},
			stop: proxias.StopReasonEndTurn,
		},
		{
			name: "GIVEN a turn cut at the limit of tokens THEN the loop ends",
			given: [][]proxias.StreamUpdate{
				{
					{
						Events: []proxias.Event{
							{
								ID:      "1",
								Type:    proxias.EventTypeMessage,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("Once upon"),
							},
						},
						StopReason: proxias.StopReasonLength,
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "2",
								Type:    proxias.EventTypeMessage,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("never"),
							},
						},
						StopReason: proxias.StopReasonEndTurn,
					},
				},
			},
			then: []string{"message hello", "message Once upon"},
			stop: proxias.StopReasonLength,
		},
		{
			name: "GIVEN a turn the model declined THEN the loop ends",
			given: [][]proxias.StreamUpdate{
				{
					{
						Events: []proxias.Event{
							{
								ID:     "1",
								Type:   proxias.EventTypeMessage,
								Source: proxias.NewAgentSource("ada"),
							},
						},
						StopReason: proxias.StopReasonContentFilter,
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "2",
								Type:    proxias.EventTypeMessage,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("never"),
							},
						},
						StopReason: proxias.StopReasonEndTurn,
					},
				},
			},
			then: []string{"message hello", "message "},
			stop: proxias.StopReasonContentFilter,
		},
		{
			name: "GIVEN a turn whose actions ran THEN the next turn reasons on their results",
			given: [][]proxias.StreamUpdate{
				{
					{
						Events: []proxias.Event{
							{
								ID:      "q",
								Type:    proxias.EventTypeActionRequest,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte(`{"action_id":"echo"}`),
							},
						},
					},
					{
						Events: []proxias.Event{
							{
								ID:          "r",
								CausationID: "q",
								Type:        proxias.EventTypeActionResult,
								Source:      proxias.NewSystemSource("actions"),
								Payload: []byte(
									`{"action_id":"echo","status":"completed"}`,
								),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "d",
								Type:    proxias.EventTypeMessage,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("Done"),
							},
						},
						StopReason: proxias.StopReasonEndTurn,
					},
				},
			},
			then: []string{
				"message hello",
				`action.request {"action_id":"echo"}`,
				`action.result {"action_id":"echo","status":"completed"}`,
				"message Done",
			},
			stop: proxias.StopReasonEndTurn,
		},
		{
			name: "GIVEN a turn that ends on a thought, without a stop reason THEN the agent goes on",
			given: [][]proxias.StreamUpdate{
				{
					{
						Events: []proxias.Event{
							{
								ID:      "1",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("Hmm"),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "2",
								Type:    proxias.EventTypeMessage,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("Got it"),
							},
						},
						StopReason: proxias.StopReasonEndTurn,
					},
				},
			},
			then: []string{
				"message hello",
				"cognition.thought Hmm",
				"message Got it",
			},
			stop: proxias.StopReasonEndTurn,
		},
		{
			name: "GIVEN turns that add nothing and never end THEN it is the strategy's bug: the loop ends at its max turns",
			then: []string{"message hello"},
			stop: proxias.StopReasonMaxTurns,
		},
		{
			name: "GIVEN turns that never end THEN the loop ends at its max turns, which is not an error",
			given: [][]proxias.StreamUpdate{
				{
					{
						Events: []proxias.Event{
							{
								ID:      "1",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("1"),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "2",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("2"),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "3",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("3"),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "4",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("4"),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "5",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("5"),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "6",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("6"),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "7",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("7"),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "8",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("8"),
							},
						},
					},
				},
				{
					{
						Events: []proxias.Event{
							{
								ID:      "9",
								Type:    proxias.EventTypeThought,
								Source:  proxias.NewAgentSource("ada"),
								Payload: []byte("9"),
							},
						},
					},
				},
			},
			then: []string{
				"message hello",
				"cognition.thought 1", "cognition.thought 2", "cognition.thought 3", "cognition.thought 4",
				"cognition.thought 5", "cognition.thought 6", "cognition.thought 7", "cognition.thought 8",
			},
			stop: proxias.StopReasonMaxTurns,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			turn := 0
			strategy := proxias.StrategyFunc(
				func(context.Context, proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
					return func(yield func(proxias.StreamUpdate) bool) {
						if turn >= len(tt.given) {
							return
						}
						turn++
						for _, u := range tt.given[turn-1] {
							if !yield(u) {
								return
							}
						}
					}
				},
			)

			stream, err := (&proxias.Agent{ID: "ada"}).Loop(
				context.Background(),
				proxias.NewSession(proxias.NewMessageEvent(proxias.NewUserSource("john"), "hello")),
				strategy,
			)
			require.NoError(t, err)
			s, err := stream.Collect()

			require.NoError(t, err)
			assert.Equal(t, tt.stop, stream.StopReason())
			got := make([]string, 0, len(s.Events()))
			for _, e := range s.Events() {
				got = append(got, string(e.Type)+" "+e.Text())
			}
			assert.Equal(t, tt.then, got)
		})
	}
}

func TestAgentWatch(t *testing.T) {
	tests := []struct {
		name  string
		given []string
		// failOn is the event whose loop fails: "*" for every one.
		failOn string
		then   []string
		err    bool
	}{
		{
			name:  "GIVEN events THEN each is appended, then looped on",
			given: []string{"93%", "95%"},
			then:  []string{"93%", "seen 93%", "95%", "seen 95%"},
		},
		{
			name:   "GIVEN every loop fails THEN their errors are streamed, and each event still comes",
			given:  []string{"93%", "95%"},
			failOn: "*",
			then:   []string{"93%", "95%"},
			err:    true,
		},
		{
			name:   "GIVEN the first event's loop fails THEN the second's still runs",
			given:  []string{"93%", "95%"},
			failOn: "93%",
			then:   []string{"93%", "95%", "seen 95%"},
			err:    true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := proxias.StrategyFunc(
				func(_ context.Context, inv proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
					return func(yield func(proxias.StreamUpdate) bool) {
						if tt.failOn == "*" || tt.failOn == inv.LastEvent().Text() {
							yield(
								proxias.StreamUpdate{Err: errors.New("down")},
							)
							return
						}
						yield(
							proxias.StreamUpdate{
								Events: []proxias.Event{
									proxias.NewEvent(
										proxias.EventTypeMessage,
										proxias.NewAgentSource("otto"),
										[]byte(
											"seen "+inv.LastEvent().
												Text(),
										),
									),
								},
								StopReason: proxias.StopReasonEndTurn,
							},
						)
					}
				},
			)
			events := make(chan proxias.Event, len(tt.given))
			for _, text := range tt.given {
				events <- proxias.NewEvent("probe.disk", proxias.NewSystemSource("db-1"), []byte(text))
			}
			close(events)

			stream, err := (&proxias.Agent{ID: "otto"}).Watch(
				context.Background(),
				proxias.NewSession(),
				events,
				strategy,
			)
			require.NoError(t, err)
			s, err := stream.Collect()
			added := s.Events()

			assert.Equal(t, tt.err, err != nil)
			got := make([]string, 0, len(added))
			for _, e := range added {
				got = append(got, e.Text())
			}
			assert.Equal(t, tt.then, got)
			assert.Len(t, s.Events(), len(tt.then))
		})
	}
}

func TestAgentMiddleware(t *testing.T) {
	tests := []struct {
		name  string
		given []string
		then  []string
	}{
		{
			name: "GIVEN no middleware THEN the strategy alone",
			then: []string{"turn"},
		},
		{
			name:  "GIVEN middleware THEN the first is the outermost",
			given: []string{"outer", "inner"},
			then:  []string{"outer", "inner", "turn"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent := proxias.Agent{ID: "ada"}
			for _, name := range tt.given {
				agent.Middleware = append(
					agent.Middleware,
					func(next proxias.CognitiveStrategy) proxias.CognitiveStrategy {
						return proxias.StrategyFunc(
							func(ctx context.Context, inv proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
								inv.Session = inv.Session.WithEvents(
									proxias.NewMessageEvent(proxias.NewSystemSource(name), name),
								)
								return next.Invoke(ctx, inv)
							},
						)
					},
				)
			}
			var seen []string
			strategy := proxias.StrategyFunc(
				func(_ context.Context, inv proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
					for _, e := range inv.Session.Events() {
						seen = append(seen, e.Text())
					}
					seen = append(seen, "turn")
					return func(func(proxias.StreamUpdate) bool) {}
				},
			)

			stream, err := agent.Invoke(context.Background(), proxias.NewSession(), strategy)
			require.NoError(t, err)
			_, err = stream.Collect()

			require.NoError(t, err)
			assert.Equal(t, tt.then, seen)
		})
	}
}

func TestAgentRejectsUpFront(t *testing.T) {
	tests := []struct {
		name string
		when func(proxias.Agent) (*proxias.SessionStream, error)
		then error
	}{
		{
			name: "WHEN invoked without a strategy THEN it fails at once",
			when: func(a proxias.Agent) (*proxias.SessionStream, error) {
				return a.Invoke(context.Background(), proxias.NewSession(), nil)
			},
			then: nil,
		},
		{
			name: "WHEN looped without a strategy THEN it fails at once",
			when: func(a proxias.Agent) (*proxias.SessionStream, error) {
				return a.Loop(context.Background(), proxias.NewSession(), nil)
			},
			then: nil,
		},
		{
			name: "WHEN watching without a strategy THEN it fails at once",
			when: func(a proxias.Agent) (*proxias.SessionStream, error) {
				return a.Watch(context.Background(), proxias.NewSession(), make(chan proxias.Event), nil)
			},
			then: nil,
		},
		{
			name: "WHEN watching no events THEN it fails at once",
			when: func(a proxias.Agent) (*proxias.SessionStream, error) {
				return a.Watch(
					context.Background(),
					proxias.NewSession(),
					nil,
					proxias.StrategyFunc(
						func(context.Context, proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
							return func(func(proxias.StreamUpdate) bool) {}
						},
					),
				)
			},
			then: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream, err := tt.when(proxias.Agent{ID: "ada"})

			require.Error(t, err)
			if tt.then != nil {
				require.ErrorIs(t, err, tt.then)
			}
			assert.Nil(t, stream)
		})
	}
}

func TestAgentLoopStopsWhenItsContextEnds(t *testing.T) {
	tests := []struct {
		name string
		// cancelAt is the turn during which ctx is cancelled.
		cancelAt int
		turns    int
	}{
		{
			name:     "GIVEN ctx cancelled during the first turn THEN no second turn starts",
			cancelAt: 1,
			turns:    1,
		},
		{
			name:     "GIVEN ctx cancelled during the second turn THEN no third turn starts",
			cancelAt: 2,
			turns:    2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			turns := 0
			strategy := proxias.StrategyFunc(
				func(context.Context, proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
					return func(yield func(proxias.StreamUpdate) bool) {
						turns++
						if turns == tt.cancelAt {
							cancel()
						}
						yield(
							proxias.StreamUpdate{
								Events: []proxias.Event{
									proxias.NewEvent(
										proxias.EventTypeThought,
										proxias.NewAgentSource("ada"),
										[]byte("Hmm"),
									),
								},
							},
						)
					}
				},
			)

			stream, err := (&proxias.Agent{ID: "ada"}).Loop(
				ctx,
				proxias.NewSession(proxias.NewMessageEvent(proxias.NewUserSource("john"), "hello")),
				strategy,
			)
			require.NoError(t, err)
			_, _ = stream.Collect()

			assert.Equal(t, tt.turns, turns)
		})
	}
}
