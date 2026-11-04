package proxias_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias"
)

func TestSessionWithEvents(t *testing.T) {
	tests := []struct {
		name    string
		given   []proxias.Event
		when    []proxias.Event
		parents []proxias.EventID
	}{
		{
			name:    "GIVEN an empty session WHEN events are added THEN each follows the one before it",
			when:    []proxias.Event{{ID: "1"}, {ID: "2"}},
			parents: []proxias.EventID{"", "1"},
		},
		{
			name:    "GIVEN a session WHEN an event is added THEN it follows the last",
			given:   []proxias.Event{{ID: "1"}},
			when:    []proxias.Event{{ID: "2"}},
			parents: []proxias.EventID{"", "1"},
		},
		{
			name:    "GIVEN a session WHEN an event that has a parent is added THEN it keeps it",
			given:   []proxias.Event{{ID: "1"}, {ID: "2"}},
			when:    []proxias.Event{{ID: "3", ParentID: "1"}},
			parents: []proxias.EventID{"", "1", "1"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := proxias.NewSessionWithID("s", tt.given...)

			after := before.WithEvents(tt.when...)

			parents := make([]proxias.EventID, 0, len(after.Events()))
			for _, e := range after.Events() {
				parents = append(parents, e.ParentID)
			}
			assert.Equal(t, tt.parents, parents)
			assert.Equal(t, "s", after.ID())
			assert.Len(t, before.Events(), len(tt.given), "the session it is called on is left as it was")
		})
	}
}

func TestSessionIsImmutable(t *testing.T) {
	tests := []struct {
		name  string
		given []proxias.Event
		// when derives two sessions from the given one, and changes what it can
		// reach of them.
		when func(parent proxias.Session) (proxias.Session, proxias.Session)
		// then are the texts of the parent's events, then of each derived
		// session's.
		then [][]string
	}{
		{
			name:  "GIVEN a session WHEN two sessions are derived from it THEN they do not share their events, nor change it",
			given: []proxias.Event{{ID: "1", Payload: []byte("a")}},
			when: func(parent proxias.Session) (proxias.Session, proxias.Session) {
				return parent.WithEvents(
						proxias.Event{ID: "2", Payload: []byte("b")},
					), parent.WithEvents(
						proxias.Event{ID: "3", Payload: []byte("c")},
					)
			},
			then: [][]string{{"a"}, {"a", "b"}, {"a", "c"}},
		},
		{
			name: "GIVEN a session with room to grow WHEN two sessions are derived from it THEN neither writes over the other",
			given: []proxias.Event{
				{
					ID:      "1",
					Payload: []byte("a"),
				}, {ID: "2", Payload: []byte("b")}, {ID: "3", Payload: []byte("c")},
			},
			when: func(parent proxias.Session) (proxias.Session, proxias.Session) {
				grown := parent.WithEvents(
					proxias.Event{ID: "4", Payload: []byte("d")},
				)
				return grown.WithEvents(
						proxias.Event{ID: "5", Payload: []byte("e")},
					), grown.WithEvents(
						proxias.Event{ID: "6", Payload: []byte("f")},
					)
			},
			then: [][]string{
				{"a", "b", "c"},
				{"a", "b", "c", "d", "e"},
				{"a", "b", "c", "d", "f"},
			},
		},
		{
			name:  "GIVEN a session WHEN the events it gave are changed THEN it is not",
			given: []proxias.Event{{ID: "1", Payload: []byte("a")}},
			when: func(parent proxias.Session) (proxias.Session, proxias.Session) {
				parent.Events()[0] = proxias.Event{
					ID:      "x",
					Payload: []byte("x"),
				}
				derived := parent.WithEvents(
					proxias.Event{ID: "2", Payload: []byte("b")},
				)
				derived.Events()[1] = proxias.Event{
					ID:      "y",
					Payload: []byte("y"),
				}
				return derived, derived
			},
			then: [][]string{{"a"}, {"a", "b"}, {"a", "b"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parent := proxias.NewSession(tt.given...)

			first, second := tt.when(parent)

			got := make([][]string, 0, 3)
			for _, s := range []proxias.Session{parent, first, second} {
				texts := make([]string, 0, len(s.Events()))
				for _, e := range s.Events() {
					texts = append(texts, e.Text())
				}
				got = append(got, texts)
			}
			assert.Equal(t, tt.then, got)
		})
	}
}

func TestSessionLastEvent(t *testing.T) {
	tests := []struct {
		name  string
		given []proxias.Event
		then  proxias.EventID
	}{
		{name: "GIVEN no event THEN it is the zero event", then: ""},
		{
			name:  "GIVEN events THEN it is the last",
			given: []proxias.Event{{ID: "1"}, {ID: "2"}},
			then:  "2",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.then, proxias.NewSession(tt.given...).LastEvent().ID)
		})
	}
}
