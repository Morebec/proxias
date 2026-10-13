package openai_test

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

func TestWorkflowExecutionIntentSelectorSelect(t *testing.T) {
	type expect struct {
		descs []proxias.WorkflowDescriptor
		err   assert.ErrorAssertionFunc
	}
	both := []proxias.WorkflowDescriptor{restartService, appVersion}
	tests := []struct {
		name  string
		given string // the text the model writes
		when  []proxias.WorkflowDescriptor
		then  expect
	}{
		{
			name:  "canonical",
			given: `{"workflows":["restart_service"]}`,
			when:  both,
			then:  expect{descs: []proxias.WorkflowDescriptor{restartService}, err: assert.NoError},
		},
		{
			name:  "GIVEN the model lists them in another order THEN they keep the order they were given",
			given: `{"workflows":["app_version","restart_service"]}`,
			when:  both,
			then:  expect{descs: both, err: assert.NoError},
		},
		{
			name:  "GIVEN an ID that was not offered THEN it is ignored",
			given: `{"workflows":["rm_rf"]}`,
			when:  both,
			then:  expect{err: assert.NoError},
		},
		{
			name:  "GIVEN no workflow offered THEN the API is not called and nothing is selected",
			given: `{"workflows":["restart_service"]}`,
			when:  nil,
			then:  expect{err: assert.NoError},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, got := mockOpenAI(t, http.StatusOK, answerWith(tt.given))
			selector := openai.NewWorkflowExecutionIntentSelector(client, someModel)

			selected, err := selector.Select(t.Context(), userSays("restart api"), tt.when)

			tt.then.err(t, err)
			assert.Equal(t, tt.then.descs, selected)
			if len(tt.when) == 0 {
				assert.Empty(t, got.body, "the API was not called")
			}
		})
	}
}

func TestWorkflowExecutionIntentSelectorSelectSendsTheWorkflowsWithoutTheirSchemas(t *testing.T) {
	// Given
	client, got := mockOpenAI(t, http.StatusOK, answerWith(`{"workflows":[]}`))
	selector := openai.NewWorkflowExecutionIntentSelector(client, someModel)

	// When
	_, err := selector.Select(t.Context(), userSays("restart api"), []proxias.WorkflowDescriptor{restartService})

	// Then
	require.NoError(t, err)
	input := sentInput(t, got)
	assert.Contains(t, input, `{"id":"restart_service","name":"Restart a service","description":"Restarts a service"}`)
	assert.NotContains(t, input, "schema", "the selector does not see the parameters")
}

func TestWorkflowExecutionIntentSelectorSelectReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockOpenAI(t, http.StatusInternalServerError, `{"error":{"type":"server_error","message":"down"}}`)
	selector := openai.NewWorkflowExecutionIntentSelector(client, someModel)

	// When
	_, err := selector.Select(t.Context(), userSays("restart api"), []proxias.WorkflowDescriptor{restartService})

	// Then
	assert.ErrorContains(t, err, "openai: select workflows")
}
