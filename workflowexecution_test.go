package proxias_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

type mockWorkflowExecutionEngine struct {
	descriptorResults []proxias.WorkflowDescriptor
	executionResults  []proxias.WorkflowExecutionResult
	describeErr       error
	executeErr        error
}

func (e mockWorkflowExecutionEngine) WorkflowDescriptors(context.Context, proxias.ChatCompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return e.descriptorResults, e.describeErr
}

func (e mockWorkflowExecutionEngine) ExecuteWorkflows(context.Context, proxias.ChatCompletionRequest, []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
	return e.executionResults, e.executeErr
}

func mockWorkflowExecutionSynthesizer(resp proxias.ChatCompletionResponse, err error) proxias.WorkflowExecutionSynthesizer {
	return proxias.WorkflowExecutionSynthesizerFunc(func(context.Context, proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
		return resp, err
	})
}

func recordingWorkflowExecutionSynthesizer(recorded *proxias.WorkflowExecutionSynthesis) proxias.WorkflowExecutionSynthesizer {
	return proxias.WorkflowExecutionSynthesizerFunc(func(_ context.Context, synthesis proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
		*recorded = synthesis
		return someResponse, nil
	})
}

// recordingNext is the rest of a chain: it records that it was called and the workflow descriptors of its
// context, and answers.
func recordingNext(called *bool, descs *[]proxias.WorkflowDescriptor) proxias.ProcessFunc {
	return func(ctx context.Context, _ proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
		*called = true
		*descs = proxias.WorkflowDescriptorsFrom(ctx)
		return someResponse, nil
	}
}

func TestWorkflowExecutionSynthesizerFuncSynthesize(t *testing.T) {
	var synthesized proxias.WorkflowExecutionSynthesis
	var synthesizer proxias.WorkflowExecutionSynthesizer = proxias.WorkflowExecutionSynthesizerFunc(
		func(_ context.Context, synthesis proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
			synthesized = synthesis
			return someResponse, nil
		})
	synthesis := proxias.WorkflowExecutionSynthesis{
		Request:          proxias.ChatCompletionRequest{Model: "m"},
		Intent:           proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}},
		ExecutionResults: []proxias.WorkflowExecutionResult{{WorkflowID: "some_workflow", Status: "completed"}},
	}

	got, err := synthesizer.Synthesize(context.Background(), synthesis)

	require.NoError(t, err)
	assert.Equal(t, synthesis, synthesized)
	assert.Equal(t, someResponse, got)
}

func TestWorkflowExecutionSynthesizerFuncSynthesizeReturnsTheFunctionsError(t *testing.T) {
	var synthesizer proxias.WorkflowExecutionSynthesizer = proxias.WorkflowExecutionSynthesizerFunc(
		func(context.Context, proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
			return proxias.ChatCompletionResponse{}, assert.AnError
		})

	got, err := synthesizer.Synthesize(context.Background(), proxias.WorkflowExecutionSynthesis{})

	assert.ErrorIs(t, err, assert.AnError)
	assert.Zero(t, got)
}

func TestNewWorkflowInterceptionMiddleware(t *testing.T) {
	type assumptions struct {
		classifier  proxias.IntentClassifier
		engine      proxias.WorkflowExecutionEngine
		synthesizer proxias.WorkflowExecutionSynthesizer
	}
	classifier := mockClassifier(proxias.Intent{}, nil)
	engine := mockWorkflowExecutionEngine{}
	synthesizer := mockWorkflowExecutionSynthesizer(someResponse, nil)
	tests := []struct {
		name        string
		given       assumptions
		errContains string
	}{
		{
			name:  "canonical",
			given: assumptions{classifier: classifier, engine: engine, synthesizer: synthesizer},
		},
		{
			name:        "GIVEN no classifier THEN it fails",
			given:       assumptions{engine: engine, synthesizer: synthesizer},
			errContains: "needs an IntentClassifier",
		},
		{
			name:        "GIVEN no engine THEN it fails",
			given:       assumptions{classifier: classifier, synthesizer: synthesizer},
			errContains: "needs a WorkflowExecutionEngine",
		},
		{
			name:        "GIVEN no synthesizer THEN it fails",
			given:       assumptions{classifier: classifier, engine: engine},
			errContains: "needs a WorkflowExecutionSynthesizer",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := proxias.NewWorkflowInterceptionMiddleware(tt.given.classifier, tt.given.engine, tt.given.synthesizer)

			if tt.errContains != "" {
				assert.ErrorContains(t, err, tt.errContains)
				assert.Nil(t, got)
				return
			}
			require.NoError(t, err)
			assert.NotNil(t, got)
		})
	}
}

func TestWorkflowInterceptionMiddlewareRunsTheWorkflows(t *testing.T) {
	type assumptions struct {
		classifier     proxias.IntentClassifier
		workflowEngine mockWorkflowExecutionEngine
		synthesizer    proxias.WorkflowExecutionSynthesizer
	}
	type expect struct {
		resultCount int
		errContains string
	}
	tests := []struct {
		name  string
		given assumptions
		then  expect
	}{
		{
			name: "canonical",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
					executionResults:  []proxias.WorkflowExecutionResult{{WorkflowID: "some_workflow", Status: "completed"}},
				},
			},
			then: expect{
				resultCount: 1,
			},
		},
		{
			name: "GIVEN the classifier asks for several workflows THEN the synthesizer gets all the results",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent, someOtherWorkflowExecutionIntent}}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor, someOtherWorkflowDescriptor},
					executionResults: []proxias.WorkflowExecutionResult{
						{WorkflowID: "some_workflow", Status: "completed"},
						{WorkflowID: "some_other_workflow", Status: "completed"},
					},
				},
			},
			then: expect{
				resultCount: 2,
			},
		},
		{
			name: "GIVEN a workflow that failed THEN its result is synthesized, not returned as an error",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
					executionResults:  []proxias.WorkflowExecutionResult{{WorkflowID: "some_workflow", Status: "failed"}},
				},
			},
			then: expect{
				resultCount: 1,
			},
		},
		{
			name: "GIVEN the engine fails to describe THEN the error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					describeErr: assert.AnError,
				},
			},
			then: expect{
				errContains: "list workflow descriptors",
			},
		},
		{
			name: "GIVEN the classifier fails THEN the error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{}, assert.AnError),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
				},
			},
			then: expect{
				errContains: "classify intent",
			},
		},
		{
			name: "GIVEN the engine fails to execute THEN the error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
					executeErr:        assert.AnError,
				},
			},
			then: expect{
				errContains: "execute workflows",
			},
		},
		{
			name: "GIVEN the engine returns a result per execution but one THEN the error says so",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
				},
			},
			then: expect{
				errContains: "got 0 results for 1",
			},
		},
		{
			name: "GIVEN the synthesizer fails THEN the error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
					executionResults:  []proxias.WorkflowExecutionResult{{WorkflowID: "some_workflow", Status: "completed"}},
				},
				synthesizer: mockWorkflowExecutionSynthesizer(proxias.ChatCompletionResponse{}, assert.AnError),
			},
			then: expect{
				errContains: "synthesize",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var recorded proxias.WorkflowExecutionSynthesis
			synthesizer := tt.given.synthesizer
			if synthesizer == nil {
				synthesizer = recordingWorkflowExecutionSynthesizer(&recorded)
			}
			middleware, err := proxias.NewWorkflowInterceptionMiddleware(tt.given.classifier, tt.given.workflowEngine, synthesizer)
			require.NoError(t, err)
			var nextCalled bool
			var nextDescs []proxias.WorkflowDescriptor

			got, err := middleware(recordingNext(&nextCalled, &nextDescs))(t.Context(), someRequest)

			assert.False(t, nextCalled)
			if tt.then.errContains != "" {
				assert.ErrorContains(t, err, tt.then.errContains)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, someResponse, got)
			assert.Equal(t, someRequest, recorded.Request)
			assert.Len(t, recorded.ExecutionResults, tt.then.resultCount)
		})
	}
}

func TestWorkflowInterceptionMiddlewarePassesOnARequestThatAsksForNoWorkflow(t *testing.T) {
	type assumptions struct {
		classifier     proxias.IntentClassifier
		workflowEngine mockWorkflowExecutionEngine
	}
	tests := []struct {
		name  string
		given assumptions
		then  []proxias.WorkflowDescriptor
	}{
		{
			name: "canonical",
			given: assumptions{
				classifier:     mockClassifier(proxias.Intent{}, nil),
				workflowEngine: mockWorkflowExecutionEngine{descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor}},
			},
			then: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
		},
		{
			// The classifier fails if it is called: there is nothing to choose amongst.
			name: "GIVEN the engine describes no workflow THEN the classifier is not called",
			given: assumptions{
				classifier:     mockClassifier(proxias.Intent{}, assert.AnError),
				workflowEngine: mockWorkflowExecutionEngine{},
			},
			then: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			middleware, err := proxias.NewWorkflowInterceptionMiddleware(tt.given.classifier, tt.given.workflowEngine, mockWorkflowExecutionSynthesizer(someResponse, nil))
			require.NoError(t, err)
			var nextCalled bool
			var nextDescs []proxias.WorkflowDescriptor

			got, err := middleware(recordingNext(&nextCalled, &nextDescs))(t.Context(), someRequest)

			require.NoError(t, err)
			assert.Equal(t, someResponse, got)
			assert.True(t, nextCalled)
			assert.Equal(t, tt.then, nextDescs)
		})
	}
}

// cancellingWorkflowExecutionEngine runs the workflows past the end of the request: it cancels the context, and
// answers its results anyway.
type cancellingWorkflowExecutionEngine struct {
	mockWorkflowExecutionEngine
	cancel context.CancelFunc
}

func (e cancellingWorkflowExecutionEngine) ExecuteWorkflows(context.Context, proxias.ChatCompletionRequest, []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
	e.cancel()
	return e.executionResults, nil
}

func TestWorkflowInterceptionMiddlewareDoesNotSynthesizeAfterTheContextEnds(t *testing.T) {
	// Given workflows that run past the end of the request
	ctx, cancel := context.WithCancel(t.Context())
	engine := cancellingWorkflowExecutionEngine{
		mockWorkflowExecutionEngine: mockWorkflowExecutionEngine{
			descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
			executionResults:  []proxias.WorkflowExecutionResult{{WorkflowID: "some_workflow"}},
		},
		cancel: cancel,
	}
	var recorded proxias.WorkflowExecutionSynthesis
	middleware, err := proxias.NewWorkflowInterceptionMiddleware(
		mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
		engine,
		recordingWorkflowExecutionSynthesizer(&recorded),
	)
	require.NoError(t, err)

	// When
	_, err = middleware(nil)(ctx, someRequest)

	// Then it is the context's error, and nothing was synthesized
	assert.Equal(t, context.Canceled, err)
	assert.Zero(t, recorded)
}

func TestWorkflowDescriptorsFrom(t *testing.T) {
	tests := []struct {
		name string
		when context.Context
		then []proxias.WorkflowDescriptor
	}{
		{
			name: "canonical",
			when: proxias.WithWorkflowDescriptors(context.Background(), []proxias.WorkflowDescriptor{someWorkflowDescriptor}),
			then: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
		},
		{
			name: "WHEN the context carries no descriptors THEN there are none",
			when: context.Background(),
			then: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.then, proxias.WorkflowDescriptorsFrom(tt.when))
		})
	}
}
