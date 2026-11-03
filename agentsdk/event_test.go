package agentsdk_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias/agentsdk"
)

func TestEventSourceString(t *testing.T) {
	tests := []struct {
		name  string
		given agentsdk.EventSource
		then  string
	}{
		{
			name:  "GIVEN a source THEN it is its type and ID",
			given: agentsdk.NewUserSource("john"),
			then:  "user:john",
		},
		{
			name:  "GIVEN a source with metadata THEN the metadata follows",
			given: agentsdk.EventSource{Type: agentsdk.EventSourceTypeSystem, ID: "db-1", Metadata: "disk"},
			then:  "system:db-1:disk",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.then, tt.given.String())
		})
	}
}

func TestNewMessageEvent(t *testing.T) {
	e := agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hello")

	assert.NotEmpty(t, e.ID)
	assert.False(t, e.Date.IsZero())
	assert.Equal(t, agentsdk.EventTypeMessage, e.Type)
	assert.Equal(t, agentsdk.NewUserSource("john"), e.Source)
	assert.Equal(t, "hello", e.Text())
}
