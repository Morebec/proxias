package anthropic_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
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
			name:  "GIVEN no workflow THEN nothing is selected",
			given: `{"workflows":[]}`,
			when:  both,
			then:  expect{err: assert.NoError},
		},
		{
			name:  "GIVEN an answer that is not JSON THEN it fails",
			given: "none",
			when:  both,
			then:  expect{err: assert.Error},
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
			client, sent := mockAnthropic(t, 200, answerWith(tt.given))
			selector := anthropic.NewWorkflowExecutionIntentSelector(client, someModel)

			got, err := selector.Select(t.Context(), userSays("restart api"), tt.when)

			tt.then.err(t, err)
			assert.Equal(t, tt.then.descs, got)
			if len(tt.when) == 0 {
				assert.Empty(t, *sent, "the API was not called")
			}
		})
	}
}

func TestWorkflowExecutionIntentSelectorSelectSendsTheWorkflowsWithoutTheirSchemas(t *testing.T) {
	// Given
	client, sent := mockAnthropic(t, 200, answerWith(`{"workflows":[]}`))
	selector := anthropic.NewWorkflowExecutionIntentSelector(client, someModel)

	// When
	_, err := selector.Select(t.Context(), userSays("restart api"), []proxias.WorkflowDescriptor{restartService})

	// Then
	require.NoError(t, err)
	var call struct {
		Messages []struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal([]byte(*sent), &call))
	user := call.Messages[0].Content[0].Text
	assert.Contains(t, user, `{"id":"restart_service","name":"Restart a service","description":"Restarts a service"}`)
	assert.NotContains(t, user, "schema", "the selector does not see the parameters")
	assert.NotContains(t, user, "required", "the selector does not see the parameters")
}

func TestWorkflowExecutionIntentSelectorSelectReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockAnthropic(t, 500, `{"type":"error","error":{"type":"api_error","message":"down"}}`)
	selector := anthropic.NewWorkflowExecutionIntentSelector(client, someModel)

	// When
	_, err := selector.Select(t.Context(), userSays("restart api"), []proxias.WorkflowDescriptor{restartService})

	// Then
	assert.ErrorContains(t, err, "anthropic: select workflows")
}
