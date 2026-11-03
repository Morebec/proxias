package agentsdk_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias/agentsdk"
)

func TestInvocationLastEvent(t *testing.T) {
	tests := []struct {
		name  string
		given []agentsdk.Event
		then  agentsdk.EventID
	}{
		{
			name: "GIVEN a session of no event THEN it is the zero event",
			then: "",
		},
		{
			name: "GIVEN a session of events THEN it is its last",
			given: []agentsdk.Event{
				{ID: "1"},
				{ID: "2"},
			},
			then: "2",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inv := agentsdk.Invocation{
				Session: agentsdk.NewSession(tt.given...),
			}

			assert.Equal(t, tt.then, inv.LastEvent().ID)
		})
	}
}
