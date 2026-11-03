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

func TestSessionStreamUpdates(t *testing.T) {
	tests := []struct {
		name  string
		given []agentsdk.StreamUpdate
		err   error
		// then are the updates read; session, the texts of the stream's session once they are read to their end.
		then    []string
		session []string
	}{
		{
			name: "GIVEN updates THEN they are read as sent, and the session holds their events",
			given: []agentsdk.StreamUpdate{
				{Chunk: &agentsdk.StreamChunk{EventID: "a", Data: []byte("Hi")}},
				{Events: []agentsdk.Event{{ID: "a", Type: agentsdk.EventTypeThought, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("Hi"), Date: time.Now()}}},
				{Events: []agentsdk.Event{{ID: "b", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("Bye"), Date: time.Now()}}},
			},
			then:    []string{"chunk Hi", "event Hi", "event Bye"},
			session: []string{"hello", "Hi", "Bye"},
		},
		{
			name: "GIVEN an error THEN the last update carries it, and the session the events before it",
			given: []agentsdk.StreamUpdate{
				{Events: []agentsdk.Event{{ID: "a", Type: agentsdk.EventTypeThought, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("Hi"), Date: time.Now()}}},
			},
			err:     errors.New("down"),
			then:    []string{"event Hi", "error down"},
			session: []string{"hello", "Hi"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := agentsdk.StrategyFunc(func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
				return func(yield func(agentsdk.StreamUpdate) bool) {
					for _, u := range tt.given {
						if !yield(u) {
							return
						}
					}
					if tt.err != nil {
						yield(agentsdk.StreamUpdate{Err: tt.err})
					}
				}
			})
			stream, err := (&agentsdk.Agent{ID: "ada"}).Invoke(context.Background(), agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hello")), strategy)
			require.NoError(t, err)

			var got []string
			for u := range stream.Updates() {
				switch {
				case u.Err != nil:
					got = append(got, "error "+u.Err.Error())
				case u.Chunk != nil:
					got = append(got, "chunk "+string(u.Chunk.Data))
				default:
					got = append(got, "event "+u.Events[0].Text())
				}
			}

			assert.Equal(t, tt.then, got)
			texts := make([]string, 0, len(stream.Session().Events()))
			for _, e := range stream.Session().Events() {
				texts = append(texts, e.Text())
			}
			assert.Equal(t, tt.session, texts)
		})
	}
}

func TestSessionStreamSessionWhileRunning(t *testing.T) {
	tests := []struct {
		name  string
		turns int
	}{
		{name: "GIVEN a loop of many turns WHEN its session is read while it runs THEN it is safe, and grows to the final one", turns: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			turn := 0
			strategy := agentsdk.StrategyFunc(func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
				return func(yield func(agentsdk.StreamUpdate) bool) {
					turn++
					var stop agentsdk.StopReason
					if turn == tt.turns {
						stop = agentsdk.StopReasonEndTurn
					}
					yield(agentsdk.StreamUpdate{Events: []agentsdk.Event{agentsdk.NewEvent(agentsdk.EventTypeThought, agentsdk.NewAgentSource("ada"), []byte("turn"))}, StopReason: stop})
				}
			})
			stream, err := (&agentsdk.Agent{ID: "ada"}).Loop(context.Background(), agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hello")), strategy)
			require.NoError(t, err)

			done := make(chan struct{})
			seen := 0
			go func() {
				defer close(done)
				for range 1000 {
					seen = max(seen, len(stream.Session().Events()))
				}
			}()
			s, err := stream.Collect()
			<-done

			require.NoError(t, err)
			assert.Len(t, s.Events(), tt.turns+1)
			assert.LessOrEqual(t, seen, tt.turns+1, "a session read while it runs is one the stream had")
		})
	}
}

func TestSessionStreamCollect(t *testing.T) {
	tests := []struct {
		name  string
		given []agentsdk.StreamUpdate
		err   error
		then  []string
		added []string
	}{
		{
			name: "GIVEN updates THEN it is the final session, the events added after the session given",
			given: []agentsdk.StreamUpdate{
				{Chunk: &agentsdk.StreamChunk{EventID: "a", Data: []byte("Hi")}},
				{Events: []agentsdk.Event{{ID: "a", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("Hi"), Date: time.Now()}}},
			},
			then:  []string{"hello", "Hi"},
			added: []string{"Hi"},
		},
		{
			name:  "GIVEN no update THEN it is the session given",
			then:  []string{"hello"},
			added: []string{},
		},
		{
			name: "GIVEN an error THEN it is the error, and the session so far",
			given: []agentsdk.StreamUpdate{
				{Events: []agentsdk.Event{{ID: "a", Type: agentsdk.EventTypeThought, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("Hi"), Date: time.Now()}}},
			},
			err:   errors.New("down"),
			then:  []string{"hello", "Hi"},
			added: []string{"Hi"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := agentsdk.StrategyFunc(func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
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
			})
			start := agentsdk.NewSessionWithID("s1", agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hello"))
			stream, err := (&agentsdk.Agent{ID: "ada"}).Invoke(context.Background(), start, strategy)
			require.NoError(t, err)

			s, err := stream.Collect()
			added := s.Events()[len(start.Events()):]

			if tt.err != nil {
				require.ErrorIs(t, err, tt.err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, "s1", s.ID())
			texts := make([]string, 0, len(s.Events()))
			for _, e := range s.Events() {
				texts = append(texts, e.Text())
			}
			assert.Equal(t, tt.then, texts)
			addedTexts := make([]string, 0, len(added))
			for _, e := range added {
				addedTexts = append(addedTexts, e.Text())
			}
			assert.Equal(t, tt.added, addedTexts)
			assert.Equal(t, s, stream.Session(), "the stream holds the session it ended with")
		})
	}
}

func TestSessionStreamMatchesTheAgentsSession(t *testing.T) {
	tests := []struct {
		name string
		// turns are what each turn adds; the last turn's session is the agent's own, as it holds it.
		turns [][]agentsdk.Event
		when  func(agentsdk.Agent, agentsdk.Session, agentsdk.CognitiveStrategy) (*agentsdk.SessionStream, error)
	}{
		{
			name: "GIVEN a loop of two turns THEN the stream's session is the agent's, same events, same parents",
			turns: [][]agentsdk.Event{
				{{ID: "t", Type: agentsdk.EventTypeThought, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("Thinking."), Date: time.Now()}},
				{{ID: "r", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("Done."), Date: time.Now()}},
			},
			when: func(a agentsdk.Agent, s agentsdk.Session, strategy agentsdk.CognitiveStrategy) (*agentsdk.SessionStream, error) {
				return a.Loop(context.Background(), s, strategy)
			},
		},
		{
			name: "GIVEN a loop whose turn ends on an action's result THEN the result is in both sessions, chained alike",
			turns: [][]agentsdk.Event{
				{
					{ID: "q", Type: agentsdk.EventTypeActionRequest, Source: agentsdk.NewAgentSource("ada"), Payload: []byte(`{"action_id":"echo"}`), Date: time.Now()},
					{ID: "a", CausationID: "q", Type: agentsdk.EventTypeActionResult, Source: agentsdk.NewSystemSource("actions"), Payload: []byte(`{"action_id":"echo","status":"completed"}`), Date: time.Now()},
				},
				{{ID: "r", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewAgentSource("ada"), Payload: []byte("Echoed."), Date: time.Now()}},
			},
			when: func(a agentsdk.Agent, s agentsdk.Session, strategy agentsdk.CognitiveStrategy) (*agentsdk.SessionStream, error) {
				return a.Loop(context.Background(), s, strategy)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				turn int
				seen agentsdk.Session
			)
			strategy := agentsdk.StrategyFunc(func(_ context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
				return func(yield func(agentsdk.StreamUpdate) bool) {
					if turn < len(tt.turns) {
						seen = inv.Session
						turn++
						var stop agentsdk.StopReason
						if turn == len(tt.turns) {
							stop = agentsdk.StopReasonEndTurn
						}
						yield(agentsdk.StreamUpdate{Events: tt.turns[turn-1], StopReason: stop})
					}
				}
			})
			stream, err := tt.when(agentsdk.Agent{ID: "ada"}, agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hello")), strategy)
			require.NoError(t, err)

			s, err := stream.Collect()

			require.NoError(t, err)
			require.Greater(t, len(s.Events()), len(seen.Events()))
			assert.Equal(t, seen.Events(), s.Events()[:len(seen.Events())], "the agent's session, as the last turn was given it, begins the stream's")
			for i, e := range s.Events()[1:] {
				assert.Equal(t, s.Events()[i].ID, e.ParentID, "each event follows the one before it")
			}
		})
	}
}
