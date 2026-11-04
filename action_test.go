package proxias_test

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func TestSimpleActionExecutor(t *testing.T) {
	tests := []struct {
		name        string
		given       map[string]proxias.SimpleAction
		when        []proxias.ActionExecutionRequest
		descriptors []string
		then        []proxias.ActionExecutionResult
		wantErr     bool
	}{
		{
			name: "GIVEN actions THEN they are described by ID, and run in order",
			given: map[string]proxias.SimpleAction{
				"b": {
					ActionDescriptor: proxias.ActionDescriptor{ID: "b"},
					RunFn: func(_ context.Context, _ proxias.Session, params map[string]any) (proxias.ActionExecutionResult, error) {
						return proxias.ActionExecutionResult{
							ActionID: "b",
							Status:   "completed",
							Data:     params,
						}, nil
					},
				},
				"a": {
					ActionDescriptor: proxias.ActionDescriptor{ID: "a"},
					RunFn: func(context.Context, proxias.Session, map[string]any) (proxias.ActionExecutionResult, error) {
						return proxias.ActionExecutionResult{
							ActionID: "a",
							Status:   "completed",
						}, nil
					},
				},
			},
			when: []proxias.ActionExecutionRequest{
				{ActionID: "b", Params: map[string]any{"x": "1"}},
				{ActionID: "a"},
			},
			descriptors: []string{"a", "b"},
			then: []proxias.ActionExecutionResult{
				{
					ActionID: "b",
					Status:   "completed",
					Data:     map[string]any{"x": "1"},
				},
				{ActionID: "a", Status: "completed"},
			},
		},
		{
			name: "GIVEN an unknown action, and one without a function THEN they fail, their results say so, and the others run",
			given: map[string]proxias.SimpleAction{
				"a": {ActionDescriptor: proxias.ActionDescriptor{ID: "a"}},
				"b": {
					ActionDescriptor: proxias.ActionDescriptor{ID: "b"},
					RunFn: func(context.Context, proxias.Session, map[string]any) (proxias.ActionExecutionResult, error) {
						return proxias.ActionExecutionResult{
							ActionID: "b",
							Status:   "completed",
						}, nil
					},
				},
			},
			when: []proxias.ActionExecutionRequest{
				{ActionID: "z"},
				{ActionID: "a"},
				{ActionID: "b"},
			},
			descriptors: []string{"a", "b"},
			then: []proxias.ActionExecutionResult{
				{
					ActionID: "z",
					Status:   "failed",
					Data:     map[string]any{"error": "unknown action"},
				},
				{
					ActionID: "a",
					Status:   "failed",
					Data:     map[string]any{"error": "unknown action"},
				},
				{ActionID: "b", Status: "completed"},
			},
		},
		{
			name: "GIVEN an action that fails THEN it is an error",
			given: map[string]proxias.SimpleAction{
				"a": {
					ActionDescriptor: proxias.ActionDescriptor{ID: "a"},
					RunFn: func(context.Context, proxias.Session, map[string]any) (proxias.ActionExecutionResult, error) {
						return proxias.ActionExecutionResult{}, errors.New(
							"down",
						)
					},
				},
			},
			when:        []proxias.ActionExecutionRequest{{ActionID: "a"}},
			descriptors: []string{"a"},
			wantErr:     true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executor := proxias.SimpleActionExecutor{Actions: tt.given}

			descs, err := executor.Descriptors(
				context.Background(),
				proxias.Session{},
			)
			require.NoError(t, err)
			ids := make([]string, 0, len(descs))
			for _, d := range descs {
				ids = append(ids, d.ID)
			}
			assert.Equal(t, tt.descriptors, ids)

			results, err := executor.Execute(
				context.Background(),
				proxias.Session{},
				tt.when,
			)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.then, results)
		})
	}
}
