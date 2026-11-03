package proxias_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

type mockActionExecutionEngine struct {
	descriptorResults []proxias.ActionDescriptor
	executionResults  []proxias.ActionExecutionResult
	describeErr       error
	executeErr        error
}

func (e mockActionExecutionEngine) ActionDescriptors(context.Context, proxias.ChatCompletionRequest) ([]proxias.ActionDescriptor, error) {
	return e.descriptorResults, e.describeErr
}

func (e mockActionExecutionEngine) ExecuteActions(context.Context, proxias.ChatCompletionRequest, []proxias.ActionExecutionIntent) ([]proxias.ActionExecutionResult, error) {
	return e.executionResults, e.executeErr
}

func mockActionExecutionSynthesizer(resp proxias.ChatCompletionResponse, err error) proxias.ActionExecutionSynthesizer {
	return proxias.ActionExecutionSynthesizerFunc(func(context.Context, proxias.ActionExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
		return resp, err
	})
}

func recordingActionExecutionSynthesizer(recorded *proxias.ActionExecutionSynthesis) proxias.ActionExecutionSynthesizer {
	return proxias.ActionExecutionSynthesizerFunc(func(_ context.Context, synthesis proxias.ActionExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
		*recorded = synthesis
		return someResponse, nil
	})
}

// recordingNext is the rest of a chain: it records that it was called and the action descriptors of its
// context, and answers.
func recordingNext(called *bool, descs *[]proxias.ActionDescriptor) proxias.ProcessFunc {
	return func(ctx context.Context, _ proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
		*called = true
		*descs = proxias.ActionDescriptorsFrom(ctx)
		return someResponse, nil
	}
}

func TestActionExecutionSynthesizerFuncSynthesize(t *testing.T) {
	var synthesized proxias.ActionExecutionSynthesis
	var synthesizer proxias.ActionExecutionSynthesizer = proxias.ActionExecutionSynthesizerFunc(
		func(_ context.Context, synthesis proxias.ActionExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
			synthesized = synthesis
			return someResponse, nil
		})
	synthesis := proxias.ActionExecutionSynthesis{
		Request:          proxias.ChatCompletionRequest{Model: "m"},
		Intent:           proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}},
		ExecutionResults: []proxias.ActionExecutionResult{{ActionID: "some_action", Status: "completed"}},
	}

	got, err := synthesizer.Synthesize(context.Background(), synthesis)

	require.NoError(t, err)
	assert.Equal(t, synthesis, synthesized)
	assert.Equal(t, someResponse, got)
}

func TestActionExecutionSynthesizerFuncSynthesizeReturnsTheFunctionsError(t *testing.T) {
	var synthesizer proxias.ActionExecutionSynthesizer = proxias.ActionExecutionSynthesizerFunc(
		func(context.Context, proxias.ActionExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
			return proxias.ChatCompletionResponse{}, assert.AnError
		})

	got, err := synthesizer.Synthesize(context.Background(), proxias.ActionExecutionSynthesis{})

	assert.ErrorIs(t, err, assert.AnError)
	assert.Zero(t, got)
}

func TestNewActionInterceptionMiddleware(t *testing.T) {
	type assumptions struct {
		classifier  proxias.IntentClassifier
		engine      proxias.ActionExecutionEngine
		synthesizer proxias.ActionExecutionSynthesizer
	}
	classifier := mockClassifier(proxias.Intent{}, nil)
	engine := mockActionExecutionEngine{}
	synthesizer := mockActionExecutionSynthesizer(someResponse, nil)
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
			errContains: "needs an ActionExecutionEngine",
		},
		{
			name:        "GIVEN no synthesizer THEN it fails",
			given:       assumptions{classifier: classifier, engine: engine},
			errContains: "needs an ActionExecutionSynthesizer",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := proxias.NewActionInterceptionMiddleware(tt.given.classifier, tt.given.engine, tt.given.synthesizer)

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

func TestActionInterceptionMiddlewareRunsTheActions(t *testing.T) {
	type assumptions struct {
		classifier   proxias.IntentClassifier
		actionEngine mockActionExecutionEngine
		synthesizer  proxias.ActionExecutionSynthesizer
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
				classifier: mockClassifier(proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}}, nil),
				actionEngine: mockActionExecutionEngine{
					descriptorResults: []proxias.ActionDescriptor{someActionDescriptor},
					executionResults:  []proxias.ActionExecutionResult{{ActionID: "some_action", Status: "completed"}},
				},
			},
			then: expect{
				resultCount: 1,
			},
		},
		{
			name: "GIVEN the classifier asks for several actions THEN the synthesizer gets all the results",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent, someOtherActionExecutionIntent}}, nil),
				actionEngine: mockActionExecutionEngine{
					descriptorResults: []proxias.ActionDescriptor{someActionDescriptor, someOtherActionDescriptor},
					executionResults: []proxias.ActionExecutionResult{
						{ActionID: "some_action", Status: "completed"},
						{ActionID: "some_other_action", Status: "completed"},
					},
				},
			},
			then: expect{
				resultCount: 2,
			},
		},
		{
			name: "GIVEN an action that failed THEN its result is synthesized, not returned as an error",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}}, nil),
				actionEngine: mockActionExecutionEngine{
					descriptorResults: []proxias.ActionDescriptor{someActionDescriptor},
					executionResults:  []proxias.ActionExecutionResult{{ActionID: "some_action", Status: "failed"}},
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
				actionEngine: mockActionExecutionEngine{
					describeErr: assert.AnError,
				},
			},
			then: expect{
				errContains: "list action descriptors",
			},
		},
		{
			name: "GIVEN the classifier fails THEN the error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{}, assert.AnError),
				actionEngine: mockActionExecutionEngine{
					descriptorResults: []proxias.ActionDescriptor{someActionDescriptor},
				},
			},
			then: expect{
				errContains: "classify intent",
			},
		},
		{
			name: "GIVEN the engine fails to execute THEN the error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}}, nil),
				actionEngine: mockActionExecutionEngine{
					descriptorResults: []proxias.ActionDescriptor{someActionDescriptor},
					executeErr:        assert.AnError,
				},
			},
			then: expect{
				errContains: "execute actions",
			},
		},
		{
			name: "GIVEN the engine returns a result per execution but one THEN the error says so",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}}, nil),
				actionEngine: mockActionExecutionEngine{
					descriptorResults: []proxias.ActionDescriptor{someActionDescriptor},
				},
			},
			then: expect{
				errContains: "got 0 results for 1",
			},
		},
		{
			name: "GIVEN the synthesizer fails THEN the error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}}, nil),
				actionEngine: mockActionExecutionEngine{
					descriptorResults: []proxias.ActionDescriptor{someActionDescriptor},
					executionResults:  []proxias.ActionExecutionResult{{ActionID: "some_action", Status: "completed"}},
				},
				synthesizer: mockActionExecutionSynthesizer(proxias.ChatCompletionResponse{}, assert.AnError),
			},
			then: expect{
				errContains: "synthesize",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var recorded proxias.ActionExecutionSynthesis
			synthesizer := tt.given.synthesizer
			if synthesizer == nil {
				synthesizer = recordingActionExecutionSynthesizer(&recorded)
			}
			middleware, err := proxias.NewActionInterceptionMiddleware(tt.given.classifier, tt.given.actionEngine, synthesizer)
			require.NoError(t, err)
			var nextCalled bool
			var nextDescs []proxias.ActionDescriptor

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

func TestActionInterceptionMiddlewarePassesOnARequestThatAsksForNoAction(t *testing.T) {
	type assumptions struct {
		classifier   proxias.IntentClassifier
		actionEngine mockActionExecutionEngine
	}
	tests := []struct {
		name  string
		given assumptions
		then  []proxias.ActionDescriptor
	}{
		{
			name: "canonical",
			given: assumptions{
				classifier:   mockClassifier(proxias.Intent{}, nil),
				actionEngine: mockActionExecutionEngine{descriptorResults: []proxias.ActionDescriptor{someActionDescriptor}},
			},
			then: []proxias.ActionDescriptor{someActionDescriptor},
		},
		{
			// The classifier fails if it is called: there is nothing to choose amongst.
			name: "GIVEN the engine describes no action THEN the classifier is not called",
			given: assumptions{
				classifier:   mockClassifier(proxias.Intent{}, assert.AnError),
				actionEngine: mockActionExecutionEngine{},
			},
			then: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			middleware, err := proxias.NewActionInterceptionMiddleware(tt.given.classifier, tt.given.actionEngine, mockActionExecutionSynthesizer(someResponse, nil))
			require.NoError(t, err)
			var nextCalled bool
			var nextDescs []proxias.ActionDescriptor

			got, err := middleware(recordingNext(&nextCalled, &nextDescs))(t.Context(), someRequest)

			require.NoError(t, err)
			assert.Equal(t, someResponse, got)
			assert.True(t, nextCalled)
			assert.Equal(t, tt.then, nextDescs)
		})
	}
}

// cancellingActionExecutionEngine runs the actions past the end of the request: it cancels the context, and
// answers its results anyway.
type cancellingActionExecutionEngine struct {
	mockActionExecutionEngine
	cancel context.CancelFunc
}

func (e cancellingActionExecutionEngine) ExecuteActions(context.Context, proxias.ChatCompletionRequest, []proxias.ActionExecutionIntent) ([]proxias.ActionExecutionResult, error) {
	e.cancel()
	return e.executionResults, nil
}

func TestActionInterceptionMiddlewareDoesNotSynthesizeAfterTheContextEnds(t *testing.T) {
	// Given actions that run past the end of the request
	ctx, cancel := context.WithCancel(t.Context())
	engine := cancellingActionExecutionEngine{
		mockActionExecutionEngine: mockActionExecutionEngine{
			descriptorResults: []proxias.ActionDescriptor{someActionDescriptor},
			executionResults:  []proxias.ActionExecutionResult{{ActionID: "some_action"}},
		},
		cancel: cancel,
	}
	var recorded proxias.ActionExecutionSynthesis
	middleware, err := proxias.NewActionInterceptionMiddleware(
		mockClassifier(proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}}, nil),
		engine,
		recordingActionExecutionSynthesizer(&recorded),
	)
	require.NoError(t, err)

	// When
	_, err = middleware(nil)(ctx, someRequest)

	// Then it is the context's error, and nothing was synthesized
	assert.Equal(t, context.Canceled, err)
	assert.Zero(t, recorded)
}

func TestActionDescriptorsFrom(t *testing.T) {
	tests := []struct {
		name string
		when context.Context
		then []proxias.ActionDescriptor
	}{
		{
			name: "canonical",
			when: proxias.WithActionDescriptors(context.Background(), []proxias.ActionDescriptor{someActionDescriptor}),
			then: []proxias.ActionDescriptor{someActionDescriptor},
		},
		{
			name: "WHEN the context carries no descriptors THEN there are none",
			when: context.Background(),
			then: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.then, proxias.ActionDescriptorsFrom(tt.when))
		})
	}
}

func mockSimpleAction(desc proxias.ActionDescriptor, result proxias.ActionExecutionResult, err error) proxias.SimpleAction {
	return proxias.SimpleAction{
		ActionDescriptor: desc,
		RunFn: func(context.Context, proxias.ChatCompletionRequest, map[string]any) (proxias.ActionExecutionResult, error) {
			return result, err
		},
	}
}

// simpleActionRun is what a SimpleAction was given when it ran.
type simpleActionRun struct {
	req    proxias.ChatCompletionRequest
	params map[string]any
}

func recordingSimpleAction(desc proxias.ActionDescriptor, recorded *simpleActionRun) proxias.SimpleAction {
	return proxias.SimpleAction{
		ActionDescriptor: desc,
		RunFn: func(_ context.Context, req proxias.ChatCompletionRequest, params map[string]any) (proxias.ActionExecutionResult, error) {
			*recorded = simpleActionRun{req: req, params: params}
			return proxias.ActionExecutionResult{ActionID: desc.ID, Status: "completed"}, nil
		},
	}
}

func TestSimpleActionExecutionEngineActionDescriptors(t *testing.T) {
	tests := []struct {
		name  string
		given map[string]proxias.SimpleAction
		then  []proxias.ActionDescriptor
	}{
		{
			name: "canonical",
			given: map[string]proxias.SimpleAction{
				"some_action":       mockSimpleAction(someActionDescriptor, proxias.ActionExecutionResult{}, nil),
				"some_other_action": mockSimpleAction(someOtherActionDescriptor, proxias.ActionExecutionResult{}, nil),
			},
			then: []proxias.ActionDescriptor{someActionDescriptor, someOtherActionDescriptor},
		},
		{
			name:  "GIVEN no actions THEN it describes none",
			given: nil,
			then:  []proxias.ActionDescriptor{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := proxias.SimpleActionExecutionEngine{Actions: tt.given}

			got, err := engine.ActionDescriptors(t.Context(), someRequest)

			require.NoError(t, err)
			// The actions are a map: their order is not the engine's to keep.
			assert.ElementsMatch(t, tt.then, got)
		})
	}
}

func TestSimpleActionExecutionEngineExecuteActions(t *testing.T) {
	someResult := proxias.ActionExecutionResult{ActionID: "some_action", ExecutionID: "1", Status: "completed"}
	someOtherResult := proxias.ActionExecutionResult{ActionID: "some_other_action", ExecutionID: "2", Status: "completed"}
	actions := map[string]proxias.SimpleAction{
		"some_action":       mockSimpleAction(someActionDescriptor, someResult, nil),
		"some_other_action": mockSimpleAction(someOtherActionDescriptor, someOtherResult, nil),
	}
	tests := []struct {
		name  string
		given map[string]proxias.SimpleAction
		when  []proxias.ActionExecutionIntent
		then  []proxias.ActionExecutionResult
	}{
		{
			name:  "canonical",
			given: actions,
			when:  []proxias.ActionExecutionIntent{someActionExecutionIntent},
			then:  []proxias.ActionExecutionResult{someResult},
		},
		{
			name:  "WHEN several actions are asked for THEN there is a result per execution, in their order",
			given: actions,
			when:  []proxias.ActionExecutionIntent{someOtherActionExecutionIntent, someActionExecutionIntent},
			then:  []proxias.ActionExecutionResult{someOtherResult, someResult},
		},
		{
			name:  "WHEN an action is asked for twice THEN it runs twice",
			given: actions,
			when:  []proxias.ActionExecutionIntent{someActionExecutionIntent, someActionExecutionIntent},
			then:  []proxias.ActionExecutionResult{someResult, someResult},
		},
		{
			name: "GIVEN an action that answers a failure THEN it is its result, not an error",
			given: map[string]proxias.SimpleAction{
				"some_action": mockSimpleAction(someActionDescriptor, proxias.ActionExecutionResult{ActionID: "some_action", Status: "failed"}, nil),
			},
			when: []proxias.ActionExecutionIntent{someActionExecutionIntent},
			then: []proxias.ActionExecutionResult{{ActionID: "some_action", Status: "failed"}},
		},
		{
			name:  "WHEN no action is asked for THEN there are no results",
			given: actions,
			when:  nil,
			then:  []proxias.ActionExecutionResult{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := proxias.SimpleActionExecutionEngine{Actions: tt.given}

			got, err := engine.ExecuteActions(t.Context(), someRequest, tt.when)

			require.NoError(t, err)
			assert.Equal(t, tt.then, got)
		})
	}
}

func TestSimpleActionExecutionEngineExecuteActionsGivesTheActionTheRequestAndItsParams(t *testing.T) {
	// Given
	var recorded simpleActionRun
	engine := proxias.SimpleActionExecutionEngine{Actions: map[string]proxias.SimpleAction{
		"some_action": recordingSimpleAction(someActionDescriptor, &recorded),
	}}
	params := map[string]any{"city": "Montreal"}

	// When
	_, err := engine.ExecuteActions(t.Context(), someRequest, []proxias.ActionExecutionIntent{{ActionID: "some_action", Params: params}})

	// Then
	require.NoError(t, err)
	assert.Equal(t, simpleActionRun{req: someRequest, params: params}, recorded)
}

func TestSimpleActionExecutionEngineExecuteActionsAnswersAnUnknownActionWithAFailedResult(t *testing.T) {
	// Given an engine that does not know the action asked for
	engine := proxias.SimpleActionExecutionEngine{Actions: map[string]proxias.SimpleAction{
		"some_other_action": mockSimpleAction(someOtherActionDescriptor, proxias.ActionExecutionResult{}, nil),
	}}

	// When
	got, err := engine.ExecuteActions(t.Context(), someRequest, []proxias.ActionExecutionIntent{someActionExecutionIntent})

	// Then
	require.NoError(t, err)
	require.Len(t, got, 1)
	assert.Equal(t, "some_action", got[0].ActionID)
	assert.Equal(t, "failed", got[0].Status)
	assert.NotEmpty(t, got[0].ExecutionID)
	assert.Equal(t, map[string]any{"error": "action not found: some_action"}, got[0].Data)
}

func TestSimpleActionExecutionEngineExecuteActionsReturnsTheActionsError(t *testing.T) {
	// Given an action that fails to run
	engine := proxias.SimpleActionExecutionEngine{Actions: map[string]proxias.SimpleAction{
		"some_action": mockSimpleAction(someActionDescriptor, proxias.ActionExecutionResult{}, assert.AnError),
	}}

	// When
	_, err := engine.ExecuteActions(t.Context(), someRequest, []proxias.ActionExecutionIntent{someActionExecutionIntent})

	// Then
	assert.ErrorIs(t, err, assert.AnError)
}
