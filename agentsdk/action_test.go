package agentsdk_test

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
)

func TestSimpleActionExecutor(t *testing.T) {
	tests := []struct {
		name        string
		given       map[string]agentsdk.SimpleAction
		when        []agentsdk.ActionExecutionRequest
		descriptors []string
		then        []agentsdk.ActionExecutionResult
		wantErr     bool
	}{
		{
			name: "GIVEN actions THEN they are described by ID, and run in order",
			given: map[string]agentsdk.SimpleAction{
				"b": {ActionDescriptor: agentsdk.ActionDescriptor{ID: "b"}, RunFn: func(_ context.Context, _ agentsdk.Session, params map[string]any) (agentsdk.ActionExecutionResult, error) {
					return agentsdk.ActionExecutionResult{ActionID: "b", Status: "completed", Data: params}, nil
				}},
				"a": {ActionDescriptor: agentsdk.ActionDescriptor{ID: "a"}, RunFn: func(context.Context, agentsdk.Session, map[string]any) (agentsdk.ActionExecutionResult, error) {
					return agentsdk.ActionExecutionResult{ActionID: "a", Status: "completed"}, nil
				}},
			},
			when:        []agentsdk.ActionExecutionRequest{{ActionID: "b", Params: map[string]any{"x": "1"}}, {ActionID: "a"}},
			descriptors: []string{"a", "b"},
			then: []agentsdk.ActionExecutionResult{
				{ActionID: "b", Status: "completed", Data: map[string]any{"x": "1"}},
				{ActionID: "a", Status: "completed"},
			},
		},
		{
			name: "GIVEN an unknown action THEN it is an error",
			given: map[string]agentsdk.SimpleAction{
				"a": {ActionDescriptor: agentsdk.ActionDescriptor{ID: "a"}},
			},
			when:        []agentsdk.ActionExecutionRequest{{ActionID: "z"}},
			descriptors: []string{"a"},
			wantErr:     true,
		},
		{
			name: "GIVEN an action that fails THEN it is an error",
			given: map[string]agentsdk.SimpleAction{
				"a": {ActionDescriptor: agentsdk.ActionDescriptor{ID: "a"}, RunFn: func(context.Context, agentsdk.Session, map[string]any) (agentsdk.ActionExecutionResult, error) {
					return agentsdk.ActionExecutionResult{}, errors.New("down")
				}},
			},
			when:        []agentsdk.ActionExecutionRequest{{ActionID: "a"}},
			descriptors: []string{"a"},
			wantErr:     true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executor := agentsdk.SimpleActionExecutor{Actions: tt.given}

			descs, err := executor.Descriptors(context.Background(), agentsdk.Session{})
			require.NoError(t, err)
			ids := make([]string, 0, len(descs))
			for _, d := range descs {
				ids = append(ids, d.ID)
			}
			assert.Equal(t, tt.descriptors, ids)

			results, err := executor.Execute(context.Background(), agentsdk.Session{}, tt.when)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.then, results)
		})
	}
}
