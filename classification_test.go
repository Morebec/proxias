package proxias_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias"
)

func mockClassifier(intent proxias.Intent, err error) proxias.IntentClassifier {
	return proxias.IntentClassifierFunc(func(context.Context, proxias.ChatCompletionRequest, []proxias.ActionDescriptor) (proxias.Intent, error) {
		return intent, err
	})
}

func mockActionSelector(returnedDescriptors []proxias.ActionDescriptor, err error) proxias.ActionExecutionIntentSelector {
	return proxias.ActionExecutionIntentSelectorFunc(func(context.Context, proxias.ChatCompletionRequest, []proxias.ActionDescriptor) ([]proxias.ActionDescriptor, error) {
		return returnedDescriptors, err
	})
}

func mockParamExtractor(executions []proxias.ActionExecutionIntent, err error) proxias.ActionExecutionIntentParameterExtractor {
	return proxias.ActionExecutionIntentParameterExtractorFunc(func(context.Context, proxias.ChatCompletionRequest, []proxias.ActionDescriptor) ([]proxias.ActionExecutionIntent, error) {
		return executions, err
	})
}

var someActionDescriptor = proxias.ActionDescriptor{ID: "some_action"}
var someOtherActionDescriptor = proxias.ActionDescriptor{ID: "some_other_action"}
var someActionExecutionIntent = proxias.ActionExecutionIntent{ActionID: "some_action"}
var someOtherActionExecutionIntent = proxias.ActionExecutionIntent{ActionID: "some_other_action"}

func TestCompositeIntentClassifierClassify(t *testing.T) {
	type assumptions struct {
		selector  proxias.ActionExecutionIntentSelector
		extractor proxias.ActionExecutionIntentParameterExtractor
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
				selector: mockActionSelector([]proxias.ActionDescriptor{
					someActionDescriptor,
				}, nil),
				extractor: mockParamExtractor([]proxias.ActionExecutionIntent{someActionExecutionIntent}, nil),
			},
			then: expect{
				intent: proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}},
				err:    assert.NoError,
			},
		},
		{
			name: "WHEN selector fails THEN should return an error",
			given: assumptions{
				selector:  mockActionSelector(nil, assert.AnError),
				extractor: mockParamExtractor(nil, nil),
			},
			then: expect{intent: proxias.Intent{}, err: assert.Error},
		},
		{
			name: "WHEN extractor fails THEN should return an error",
			given: assumptions{
				selector:  mockActionSelector([]proxias.ActionDescriptor{someActionDescriptor}, nil),
				extractor: mockParamExtractor(nil, assert.AnError),
			},
			then: expect{intent: proxias.Intent{}, err: assert.Error},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			classifier := proxias.NewCompositeIntentClassifier(tt.given.selector, tt.given.extractor)

			got, err := classifier.Classify(t.Context(), proxias.ChatCompletionRequest{Model: "m"}, []proxias.ActionDescriptor{
				someActionDescriptor,
				someOtherActionDescriptor,
			})

			tt.then.err(t, err)
			assert.Equal(t, tt.then.intent, got)
		})
	}
}

func TestIntentClassifierFuncClassify(t *testing.T) {
	expectedCtx := t.Context()
	expectedReq := proxias.ChatCompletionRequest{Model: "some_model"}
	expectedDescriptors := []proxias.ActionDescriptor{
		{Name: "SomeAction"},
	}
	expectedIntent := proxias.Intent{
		ActionExecutions: []proxias.ActionExecutionIntent{
			{ActionID: "some_action_id"},
		},
	}
	expectedErr := assert.AnError

	classifier := proxias.IntentClassifierFunc(
		func(ctx context.Context, req proxias.ChatCompletionRequest, descriptors []proxias.ActionDescriptor) (proxias.Intent, error) {
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
			name:  "GIVEN an intent with an empty list of action executions THEN should return true",
			given: proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{}},
			then:  assert.True,
		},
		{
			name:  "GIVEN an intent with an action execution THEN should return false",
			given: proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}},
			then:  assert.False,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.then(t, tt.given.IsReasoningIntent())
		})
	}
}
