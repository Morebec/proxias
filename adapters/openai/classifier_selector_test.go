package openai_test

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

func TestActionExecutionIntentSelectorSelect(t *testing.T) {
	type expect struct {
		descs []proxias.ActionDescriptor
		err   assert.ErrorAssertionFunc
	}
	both := []proxias.ActionDescriptor{restartService, appVersion}
	tests := []struct {
		name  string
		given string // the text the model writes
		when  []proxias.ActionDescriptor
		then  expect
	}{
		{
			name:  "canonical",
			given: `{"actions":["restart_service"]}`,
			when:  both,
			then:  expect{descs: []proxias.ActionDescriptor{restartService}, err: assert.NoError},
		},
		{
			name:  "GIVEN the model lists them in another order THEN they keep the order they were given",
			given: `{"actions":["app_version","restart_service"]}`,
			when:  both,
			then:  expect{descs: both, err: assert.NoError},
		},
		{
			name:  "GIVEN an ID that was not offered THEN it is ignored",
			given: `{"actions":["rm_rf"]}`,
			when:  both,
			then:  expect{err: assert.NoError},
		},
		{
			name:  "GIVEN no action offered THEN the API is not called and nothing is selected",
			given: `{"actions":["restart_service"]}`,
			when:  nil,
			then:  expect{err: assert.NoError},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, got := mockOpenAI(t, http.StatusOK, answerWith(tt.given))
			selector := openai.NewActionExecutionIntentSelector(client, someModel)

			selected, err := selector.Select(t.Context(), userSays("restart api"), tt.when)

			tt.then.err(t, err)
			assert.Equal(t, tt.then.descs, selected)
			if len(tt.when) == 0 {
				assert.Empty(t, got.body, "the API was not called")
			}
		})
	}
}

func TestActionExecutionIntentSelectorSelectSendsTheActionsWithoutTheirSchemas(t *testing.T) {
	// Given
	client, got := mockOpenAI(t, http.StatusOK, answerWith(`{"actions":[]}`))
	selector := openai.NewActionExecutionIntentSelector(client, someModel)

	// When
	_, err := selector.Select(t.Context(), userSays("restart api"), []proxias.ActionDescriptor{restartService})

	// Then
	require.NoError(t, err)
	input := sentInput(t, got)
	assert.Contains(t, input, `{"id":"restart_service","name":"Restart a service","description":"Restarts a service"}`)
	assert.NotContains(t, input, "schema", "the selector does not see the parameters")
}

func TestActionExecutionIntentSelectorSelectReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockOpenAI(t, http.StatusInternalServerError, `{"error":{"type":"server_error","message":"down"}}`)
	selector := openai.NewActionExecutionIntentSelector(client, someModel)

	// When
	_, err := selector.Select(t.Context(), userSays("restart api"), []proxias.ActionDescriptor{restartService})

	// Then
	assert.ErrorContains(t, err, "openai: select actions")
}
