package proxias_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias"
)

func mockWorkflowSelector(returnedDescriptors []proxias.WorkflowDescriptor, err error) proxias.WorkflowExecutionIntentSelector {
	return proxias.WorkflowExecutionIntentSelectorFunc(func(context.Context, proxias.CompletionRequest, []proxias.WorkflowDescriptor) ([]proxias.WorkflowDescriptor, error) {
		return returnedDescriptors, err
	})
}

func mockParamExtractor(executions []proxias.WorkflowExecutionIntent, err error) proxias.WorkflowExecutionIntentParameterExtractor {
	return proxias.WorkflowExecutionIntentParameterExtractorFunc(func(context.Context, proxias.CompletionRequest, []proxias.WorkflowDescriptor) ([]proxias.WorkflowExecutionIntent, error) {
		return executions, err
	})
}

var someWorkflowDescriptor = proxias.WorkflowDescriptor{ID: "some_workflow"}
var someOtherWorkflowDescriptor = proxias.WorkflowDescriptor{ID: "some_other_workflow"}
var someWorkflowExecutionIntent = proxias.WorkflowExecutionIntent{WorkflowID: "some_workflow"}
var someOtherWorkflowExecutionIntent = proxias.WorkflowExecutionIntent{WorkflowID: "some_other_workflow"}

func TestCompositeIntentClassifierClassify(t *testing.T) {
	type assumptions struct {
		selector  proxias.WorkflowExecutionIntentSelector
		extractor proxias.WorkflowExecutionIntentParameterExtractor
	}
	type expect struct {
		intent proxias.Intent
		err    assert.ErrorAssertionFunc
	}
	tests := []struct {
		name  string
		given assumptions
		then  expect
	}{
		{
			name: "canonical",
			given: assumptions{
				selector: mockWorkflowSelector([]proxias.WorkflowDescriptor{
					someWorkflowDescriptor,
				}, nil),
				extractor: mockParamExtractor([]proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}, nil),
			},
			then: expect{
				intent: proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}},
				err:    assert.NoError,
			},
		},
		{
			name: "WHEN selector fails THEN should return an error",
			given: assumptions{
				selector:  mockWorkflowSelector(nil, assert.AnError),
				extractor: mockParamExtractor(nil, nil),
			},
			then: expect{intent: proxias.Intent{}, err: assert.Error},
		},
		{
			name: "WHEN extractor fails THEN should return an error",
			given: assumptions{
				selector:  mockWorkflowSelector([]proxias.WorkflowDescriptor{someWorkflowDescriptor}, nil),
				extractor: mockParamExtractor(nil, assert.AnError),
			},
			then: expect{intent: proxias.Intent{}, err: assert.Error},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			classifier := proxias.NewCompositeIntentClassifier(tt.given.selector, tt.given.extractor)

			got, err := classifier.Classify(t.Context(), proxias.CompletionRequest{Model: "m"}, []proxias.WorkflowDescriptor{
				someWorkflowDescriptor,
				someOtherWorkflowDescriptor,
			})

			tt.then.err(t, err)
			assert.Equal(t, tt.then.intent, got)
		})
	}
}

func TestIntentClassifierFuncClassify(t *testing.T) {
	expectedCtx := t.Context()
	expectedReq := proxias.CompletionRequest{Model: "some_model"}
	expectedDescriptors := []proxias.WorkflowDescriptor{
		{Name: "SomeWorkflow"},
	}
	expectedIntent := proxias.Intent{
		WorkflowExecutions: []proxias.WorkflowExecutionIntent{
			{WorkflowID: "some_workflow_id"},
		},
	}
	expectedErr := assert.AnError

	classifier := proxias.IntentClassifierFunc(
		func(ctx context.Context, req proxias.CompletionRequest, descriptors []proxias.WorkflowDescriptor) (proxias.Intent, error) {
			assert.Equal(t, expectedCtx, ctx)
			assert.Equal(t, expectedReq, req)
			assert.Equal(t, expectedDescriptors, descriptors)

			return expectedIntent, expectedErr
		},
	)

	// WHEN it classifies a request
	got, err := classifier.Classify(expectedCtx, expectedReq, expectedDescriptors)

	// THEN returns the expected error and intent
	assert.ErrorIs(t, err, expectedErr)
	assert.Equal(t, expectedIntent, got)
}

func TestIntentIsReasoningIntent(t *testing.T) {
	tests := []struct {
		name  string
		given proxias.Intent
		then  assert.BoolAssertionFunc
	}{
		{
			name:  "GIVEN an empty intent THEN should return true",
			given: proxias.Intent{},
			then:  assert.True,
		},
		{
			name:  "GIVEN an intent with an empty list of workflow executions THEN should return true",
			given: proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{}},
			then:  assert.True,
		},
		{
			name:  "GIVEN an intent with a workflow execution THEN should return false",
			given: proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}},
			then:  assert.False,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.then(t, tt.given.IsReasoningIntent())
		})
	}
}
