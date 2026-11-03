package agentsdk_test

import (
	"context"
	"errors"
	"iter"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
)

func TestAgentInvoke(t *testing.T) {
	tests := []struct {
		name  string
		given []agentsdk.StreamUpdate
		err   error
		then  []string
	}{
		{
			name: "GIVEN a turn that streams its answer THEN the chunk comes, then the event",
			given: []agentsdk.StreamUpdate{
				{
					Chunk: &agentsdk.StreamChunk{
						EventID: "r",
						Data:    []byte("Hi"),
					},
				},
				{
					Events: []agentsdk.Event{
						{
							ID:      "r",
							Type:    agentsdk.EventTypeMessage,
							Source:  agentsdk.NewAgentSource("ada"),
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
			given: []agentsdk.StreamUpdate{
				{
					Events: []agentsdk.Event{
						agentsdk.NewEvent(
							agentsdk.EventTypeActionRequest,
							agentsdk.NewAgentSource("ada"),
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
			strategy := agentsdk.StrategyFunc(
				func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
					return func(yield func(agentsdk.StreamUpdate) bool) {
						for _, d := range tt.given {
							if !yield(d) {
								return
							}
						}
						if tt.err != nil {
							yield(agentsdk.StreamUpdate{Err: tt.err})
						}
					}
				},
			)
			agent := agentsdk.Agent{ID: "ada"}
			start := agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hello"))

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
				assert.Equal(t, agentsdk.NewAgentSource("ada"), e.Source)
				assert.Equal(t, start.Events()[0].ID, e.ParentID)
			}
		})
	}
}

func TestAgentLoop(t *testing.T) {
	tests := []struct {
		name string
		// given are the updates of each turn.
		given [][]agentsdk.StreamUpdate
		then  []string
		stop  agentsdk.StopReason
	}{
		{
			name: "GIVEN a turn that ends its turn THEN the loop ends",
			given: [][]agentsdk.StreamUpdate{
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "1",
								Type:    agentsdk.EventTypeMessage,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("Hi"),
							},
						},
						StopReason: agentsdk.StopReasonEndTurn,
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "2",
								Type:    agentsdk.EventTypeMessage,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("never"),
							},
						},
						StopReason: agentsdk.StopReasonEndTurn,
					},
				},
			},
			then: []string{"message hello", "message Hi"},
			stop: agentsdk.StopReasonEndTurn,
		},
		{
			name: "GIVEN a turn cut at the limit of tokens THEN the loop ends",
			given: [][]agentsdk.StreamUpdate{
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "1",
								Type:    agentsdk.EventTypeMessage,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("Once upon"),
							},
						},
						StopReason: agentsdk.StopReasonLength,
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "2",
								Type:    agentsdk.EventTypeMessage,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("never"),
							},
						},
						StopReason: agentsdk.StopReasonEndTurn,
					},
				},
			},
			then: []string{"message hello", "message Once upon"},
			stop: agentsdk.StopReasonLength,
		},
		{
			name: "GIVEN a turn the model declined THEN the loop ends",
			given: [][]agentsdk.StreamUpdate{
				{
					{
						Events: []agentsdk.Event{
							{
								ID:     "1",
								Type:   agentsdk.EventTypeMessage,
								Source: agentsdk.NewAgentSource("ada"),
							},
						},
						StopReason: agentsdk.StopReasonContentFilter,
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "2",
								Type:    agentsdk.EventTypeMessage,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("never"),
							},
						},
						StopReason: agentsdk.StopReasonEndTurn,
					},
				},
			},
			then: []string{"message hello", "message "},
			stop: agentsdk.StopReasonContentFilter,
		},
		{
			name: "GIVEN a turn whose actions ran THEN the next turn reasons on their results",
			given: [][]agentsdk.StreamUpdate{
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "q",
								Type:    agentsdk.EventTypeActionRequest,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte(`{"action_id":"echo"}`),
							},
						},
					},
					{
						Events: []agentsdk.Event{
							{
								ID:          "r",
								CausationID: "q",
								Type:        agentsdk.EventTypeActionResult,
								Source:      agentsdk.NewSystemSource("actions"),
								Payload: []byte(
									`{"action_id":"echo","status":"completed"}`,
								),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "d",
								Type:    agentsdk.EventTypeMessage,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("Done"),
							},
						},
						StopReason: agentsdk.StopReasonEndTurn,
					},
				},
			},
			then: []string{
				"message hello",
				`action.request {"action_id":"echo"}`,
				`action.result {"action_id":"echo","status":"completed"}`,
				"message Done",
			},
			stop: agentsdk.StopReasonEndTurn,
		},
		{
			name: "GIVEN a turn that ends on a thought, without a stop reason THEN the agent goes on",
			given: [][]agentsdk.StreamUpdate{
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "1",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("Hmm"),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "2",
								Type:    agentsdk.EventTypeMessage,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("Got it"),
							},
						},
						StopReason: agentsdk.StopReasonEndTurn,
					},
				},
			},
			then: []string{
				"message hello",
				"cognition.thought Hmm",
				"message Got it",
			},
			stop: agentsdk.StopReasonEndTurn,
		},
		{
			name: "GIVEN turns that add nothing and never end THEN it is the strategy's bug: the loop ends at its max turns",
			then: []string{"message hello"},
			stop: agentsdk.StopReasonMaxTurns,
		},
		{
			name: "GIVEN turns that never end THEN the loop ends at its max turns, which is not an error",
			given: [][]agentsdk.StreamUpdate{
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "1",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("1"),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "2",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("2"),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "3",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("3"),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "4",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("4"),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "5",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("5"),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "6",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("6"),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "7",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("7"),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "8",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
								Payload: []byte("8"),
							},
						},
					},
				},
				{
					{
						Events: []agentsdk.Event{
							{
								ID:      "9",
								Type:    agentsdk.EventTypeThought,
								Source:  agentsdk.NewAgentSource("ada"),
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
			stop: agentsdk.StopReasonMaxTurns,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			turn := 0
			strategy := agentsdk.StrategyFunc(
				func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
					return func(yield func(agentsdk.StreamUpdate) bool) {
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

			stream, err := (&agentsdk.Agent{ID: "ada"}).Loop(
				context.Background(),
				agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hello")),
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
			strategy := agentsdk.StrategyFunc(
				func(_ context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
					return func(yield func(agentsdk.StreamUpdate) bool) {
						if tt.failOn == "*" || tt.failOn == inv.LastEvent().Text() {
							yield(
								agentsdk.StreamUpdate{Err: errors.New("down")},
							)
							return
						}
						yield(
							agentsdk.StreamUpdate{
								Events: []agentsdk.Event{
									agentsdk.NewEvent(
										agentsdk.EventTypeMessage,
										agentsdk.NewAgentSource("otto"),
										[]byte(
											"seen "+inv.LastEvent().
												Text(),
										),
									),
								},
								StopReason: agentsdk.StopReasonEndTurn,
							},
						)
					}
				},
			)
			events := make(chan agentsdk.Event, len(tt.given))
			for _, text := range tt.given {
				events <- agentsdk.NewEvent("probe.disk", agentsdk.NewSystemSource("db-1"), []byte(text))
			}
			close(events)

			stream, err := (&agentsdk.Agent{ID: "otto"}).Watch(
				context.Background(),
				agentsdk.NewSession(),
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
			agent := agentsdk.Agent{ID: "ada"}
			for _, name := range tt.given {
				agent.Middleware = append(
					agent.Middleware,
					func(next agentsdk.CognitiveStrategy) agentsdk.CognitiveStrategy {
						return agentsdk.StrategyFunc(
							func(ctx context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
								inv.Session = inv.Session.WithEvents(
									agentsdk.NewMessageEvent(agentsdk.NewSystemSource(name), name),
								)
								return next.Invoke(ctx, inv)
							},
						)
					},
				)
			}
			var seen []string
			strategy := agentsdk.StrategyFunc(
				func(_ context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
					for _, e := range inv.Session.Events() {
						seen = append(seen, e.Text())
					}
					seen = append(seen, "turn")
					return func(func(agentsdk.StreamUpdate) bool) {}
				},
			)

			stream, err := agent.Invoke(context.Background(), agentsdk.NewSession(), strategy)
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
		when func(agentsdk.Agent) (*agentsdk.SessionStream, error)
		then error
	}{
		{
			name: "WHEN invoked without a strategy THEN it fails at once",
			when: func(a agentsdk.Agent) (*agentsdk.SessionStream, error) {
				return a.Invoke(context.Background(), agentsdk.NewSession(), nil)
			},
			then: nil,
		},
		{
			name: "WHEN looped without a strategy THEN it fails at once",
			when: func(a agentsdk.Agent) (*agentsdk.SessionStream, error) {
				return a.Loop(context.Background(), agentsdk.NewSession(), nil)
			},
			then: nil,
		},
		{
			name: "WHEN watching without a strategy THEN it fails at once",
			when: func(a agentsdk.Agent) (*agentsdk.SessionStream, error) {
				return a.Watch(context.Background(), agentsdk.NewSession(), make(chan agentsdk.Event), nil)
			},
			then: nil,
		},
		{
			name: "WHEN watching no events THEN it fails at once",
			when: func(a agentsdk.Agent) (*agentsdk.SessionStream, error) {
				return a.Watch(
					context.Background(),
					agentsdk.NewSession(),
					nil,
					agentsdk.StrategyFunc(
						func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
							return func(func(agentsdk.StreamUpdate) bool) {}
						},
					),
				)
			},
			then: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream, err := tt.when(agentsdk.Agent{ID: "ada"})

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
			strategy := agentsdk.StrategyFunc(
				func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
					return func(yield func(agentsdk.StreamUpdate) bool) {
						turns++
						if turns == tt.cancelAt {
							cancel()
						}
						yield(
							agentsdk.StreamUpdate{
								Events: []agentsdk.Event{
									agentsdk.NewEvent(
										agentsdk.EventTypeThought,
										agentsdk.NewAgentSource("ada"),
										[]byte("Hmm"),
									),
								},
							},
						)
					}
				},
			)

			stream, err := (&agentsdk.Agent{ID: "ada"}).Loop(
				ctx,
				agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hello")),
				strategy,
			)
			require.NoError(t, err)
			_, _ = stream.Collect()

			assert.Equal(t, tt.turns, turns)
		})
	}
}
