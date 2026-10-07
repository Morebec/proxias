package proxias_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias"
)

func TestCompletionMessageHasToolRole(t *testing.T) {
	tests := []struct {
		name  string
		given proxias.CompletionMessage
		then  assert.BoolAssertionFunc
	}{
		{
			name:  "GIVEN a message without a role THEN it should return false",
			given: proxias.CompletionMessage{},
			then:  assert.False,
		},
		{
			name:  "GIVEN a user message THEN it should return false",
			given: proxias.CompletionMessage{Role: proxias.CompletionMessageRoleUser},
			then:  assert.False,
		},
		{
			name:  "GIVEN an assistant message THEN it should return false",
			given: proxias.CompletionMessage{Role: proxias.CompletionMessageRoleAssistant},
			then:  assert.False,
		},
		{
			name:  "GIVEN a system message THEN it should return false",
			given: proxias.CompletionMessage{Role: proxias.CompletionMessageRoleSystem},
			then:  assert.False,
		},
		{
			name:  "GIVEN a developer message THEN it should return false",
			given: proxias.CompletionMessage{Role: proxias.CompletionMessageRoleDeveloper},
			then:  assert.False,
		},
		{
			name:  "GIVEN a tool message THEN it should return true",
			given: proxias.CompletionMessage{Role: proxias.CompletionMessageRoleTool},
			then:  assert.True,
		},
		{
			name:  "GIVEN a function message, the former tool role, THEN it should return true",
			given: proxias.CompletionMessage{Role: proxias.CompletionMessageRoleFunction},
			then:  assert.True,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.then(t, tt.given.HasToolRole())
		})
	}
}

func TestCompletionRequestLastMessage(t *testing.T) {
	// The messages differ by their role only: LastMessage answers which one it is, not what it says.
	var (
		userRoleMessage      = proxias.CompletionMessage{Role: proxias.CompletionMessageRoleUser}
		assistantRoleMessage = proxias.CompletionMessage{Role: proxias.CompletionMessageRoleAssistant}
	)

	tests := []struct {
		name  string
		given []proxias.CompletionMessage
		then  proxias.CompletionMessage
	}{
		{
			name:  "GIVEN no message THEN should return no message",
			given: nil,
			then:  proxias.CompletionMessage{},
		},
		{
			name:  "GIVEN one message THEN should return that message",
			given: []proxias.CompletionMessage{userRoleMessage},
			then:  userRoleMessage,
		},
		{
			name: "GIVEN several messages THEN should return the last one",
			given: []proxias.CompletionMessage{
				userRoleMessage,
				assistantRoleMessage,
			},
			then: assistantRoleMessage,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := proxias.CompletionRequest{Messages: tt.given}

			assert.Equal(t, tt.then, req.LastMessage())
		})
	}
}
