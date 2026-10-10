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

var someRequest = proxias.ChatCompletionRequest{Model: "m"}
var someResponse = proxias.ChatCompletionResponse{Status: 200}

func mockClassifier(intent proxias.Intent, err error) proxias.IntentClassifier {
	return proxias.IntentClassifierFunc(func(context.Context, proxias.ChatCompletionRequest, []proxias.WorkflowDescriptor) (proxias.Intent, error) {
		return intent, err
	})
}

func mockWorkflowExecutionSynthesizer(resp proxias.ChatCompletionResponse, err error) proxias.WorkflowExecutionSynthesizer {
	return proxias.WorkflowExecutionSynthesizerFunc(func(context.Context, proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
		return resp, err
	})
}

func mockReasoningEngine(resp proxias.ChatCompletionResponse, err error) proxias.ReasoningEngine {
	return proxias.ReasoningEngineFunc(func(context.Context, proxias.ChatCompletionRequest, []proxias.WorkflowDescriptor) (proxias.ChatCompletionResponse, error) {
		return resp, err
	})
}

func recordingWorkflowExecutionSynthesizer(recorded *proxias.WorkflowExecutionSynthesis) proxias.WorkflowExecutionSynthesizer {
	return proxias.WorkflowExecutionSynthesizerFunc(func(_ context.Context, synthesis proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
		*recorded = synthesis
		return someResponse, nil
	})
}

func recordingReasoningEngine(recordedReq *proxias.ChatCompletionRequest, recordedDesc *[]proxias.WorkflowDescriptor) proxias.ReasoningEngine {
	return proxias.ReasoningEngineFunc(func(_ context.Context, req proxias.ChatCompletionRequest, desc []proxias.WorkflowDescriptor) (proxias.ChatCompletionResponse, error) {
		*recordedReq = req
		*recordedDesc = desc
		return someResponse, nil
	})
}

func TestControlPlaneProcessWorkflowExecutions(t *testing.T) {
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
			name: "GIVEN classifier asks for one workflow THEN synthesizer gets the result",
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
			name: "GIVEN classifier asks for multiple workflows THEN synthesizer gets all results",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{
					WorkflowExecutions: []proxias.WorkflowExecutionIntent{
						someWorkflowExecutionIntent,
						someOtherWorkflowExecutionIntent,
					},
				}, nil),
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
			name: "GIVEN workflow that failed THEN its result is synthesized, not returned as error",
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
			name: "GIVEN WorkflowExecutionEngine fails to describe THEN error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					describeErr: assert.AnError,
				},
				synthesizer: mockWorkflowExecutionSynthesizer(someResponse, nil),
			},
			then: expect{
				errContains: "list workflow descriptors",
			},
		},
		{
			name: "GIVEN classifier fails THEN error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{}, assert.AnError),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
				},
				synthesizer: mockWorkflowExecutionSynthesizer(someResponse, nil),
			},
			then: expect{
				errContains: "classify intent",
			},
		},
		{
			name: "GIVEN WorkflowExecutionEngine fails to execute THEN error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
					executeErr:        assert.AnError,
				},
				synthesizer: mockWorkflowExecutionSynthesizer(someResponse, nil),
			},
			then: expect{
				errContains: "execute workflows",
			},
		},
		{
			name: "GIVEN WorkflowExecutionSynthesizer fails THEN error says which stage failed",
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
		{
			name: "GIVEN WorkflowExecutionEngine returns wrong number of results THEN error says count mismatch",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
					executionResults:  []proxias.WorkflowExecutionResult{},
				},
				synthesizer: mockWorkflowExecutionSynthesizer(someResponse, nil),
			},
			then: expect{
				errContains: "got 0 results for 1",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synthesizer := tt.given.synthesizer
			var recorded proxias.WorkflowExecutionSynthesis

			// Use recording synthesizer for success cases, given synthesizer for error cases
			if synthesizer == nil {
				synthesizer = recordingWorkflowExecutionSynthesizer(&recorded)
			}

			controlPlane := newControlPlane(t,
				tt.given.classifier,
				tt.given.workflowEngine,
				synthesizer,
				mockReasoningEngine(someResponse, nil),
			)

			got, err := controlPlane.Process(context.Background(), someRequest)

			if tt.then.errContains != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.then.errContains)
			} else {
				require.NoError(t, err)
				assert.Equal(t, someResponse, got)
				assert.Len(t, recorded.ExecutionResults, tt.then.resultCount)
			}
		})
	}
}

func TestControlPlaneProcessReasons(t *testing.T) {
	type assumptions struct {
		classifier      proxias.IntentClassifier
		workflowEngine  mockWorkflowExecutionEngine
		reasoningEngine proxias.ReasoningEngine
	}
	type expect struct {
		descriptorCount int
		errContains     string
	}
	tests := []struct {
		name  string
		given assumptions
		then  expect
	}{
		{
			name: "GIVEN classifier asks for no workflow THEN ReasoningEngine answers",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
				},
			},
			then: expect{
				descriptorCount: 1,
			},
		},
		{
			name: "GIVEN WorkflowExecutionEngine describes no workflows THEN ReasoningEngine answers without classifying",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{},
				},
			},
			then: expect{
				descriptorCount: 0,
			},
		},
		{
			name: "GIVEN ReasoningEngine fails THEN error says which stage failed",
			given: assumptions{
				classifier: mockClassifier(proxias.Intent{}, nil),
				workflowEngine: mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
				},
				reasoningEngine: mockReasoningEngine(proxias.ChatCompletionResponse{}, assert.AnError),
			},
			then: expect{
				errContains: "reason",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var recordedReq proxias.ChatCompletionRequest
			var recordedDesc []proxias.WorkflowDescriptor

			reasoningEngine := tt.given.reasoningEngine
			if reasoningEngine == nil {
				reasoningEngine = recordingReasoningEngine(&recordedReq, &recordedDesc)
			}

			controlPlane := newControlPlane(t,
				tt.given.classifier,
				tt.given.workflowEngine,
				mockWorkflowExecutionSynthesizer(someResponse, nil),
				reasoningEngine,
			)

			got, err := controlPlane.Process(context.Background(), someRequest)

			if tt.then.errContains != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.then.errContains)
			} else {
				require.NoError(t, err)
				assert.Equal(t, someRequest, recordedReq)
				assert.Len(t, recordedDesc, tt.then.descriptorCount)
				assert.NotNil(t, got)
			}
		})
	}
}

func TestControlPlaneProcessReturnsContextErrorBeforeProcessing(t *testing.T) {
	// GIVEN a context already cancelled
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	controlPlane := newControlPlane(t,
		mockClassifier(proxias.Intent{}, nil),
		mockWorkflowExecutionEngine{descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor}},
		mockWorkflowExecutionSynthesizer(someResponse, nil),
		mockReasoningEngine(someResponse, nil),
	)

	// WHEN the request is processed
	_, err := controlPlane.Process(cancelled, someRequest)

	// THEN it is the context's error, before any stage ran
	assert.ErrorIs(t, err, context.Canceled)
}

func TestControlPlaneProcessReturnsContextErrorDuringWorkflowExecution(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	signalled := make(chan struct{})

	controlPlane := newControlPlane(t,
		mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
		&delayedWorkflowExecutionEngine{
			descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
			results:           []proxias.WorkflowExecutionResult{{WorkflowID: "some_workflow"}},
			signal:            signalled,
		},
		proxias.WorkflowExecutionSynthesizerFunc(func(context.Context, proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
			panic("should not synthesize after cancel")
		}),
		mockReasoningEngine(someResponse, nil),
	)

	go func() {
		<-signalled
		cancel()
	}()
	_, err := controlPlane.Process(ctx, someRequest)

	assert.ErrorIs(t, err, context.Canceled)
}

type delayedWorkflowExecutionEngine struct {
	descriptorResults []proxias.WorkflowDescriptor
	results           []proxias.WorkflowExecutionResult
	signal            chan struct{}
}

func (e *delayedWorkflowExecutionEngine) WorkflowDescriptors(context.Context, proxias.ChatCompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return e.descriptorResults, nil
}

func (e *delayedWorkflowExecutionEngine) ExecuteWorkflows(ctx context.Context, _ proxias.ChatCompletionRequest, _ []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
	e.signal <- struct{}{}
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestControlPlaneProcessDoesNotRecoverPanics(t *testing.T) {
	const panicValue = "boom"

	tests := []struct {
		name  string
		given proxias.ControlPlane
	}{
		{
			name: "GIVEN IntentClassifier panics THEN panic goes through",
			given: newControlPlane(t,
				proxias.IntentClassifierFunc(func(context.Context, proxias.ChatCompletionRequest, []proxias.WorkflowDescriptor) (proxias.Intent, error) {
					panic(panicValue)
				}),
				mockWorkflowExecutionEngine{descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor}},
				mockWorkflowExecutionSynthesizer(someResponse, nil),
				mockReasoningEngine(someResponse, nil),
			),
		},
		{
			name: "GIVEN WorkflowExecutionEngine panics executing THEN panic goes through",
			given: newControlPlane(t,
				mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
				&panicWorkflowExecutionEngine{panicMsg: panicValue},
				mockWorkflowExecutionSynthesizer(someResponse, nil),
				mockReasoningEngine(someResponse, nil),
			),
		},
		{
			name: "GIVEN WorkflowExecutionSynthesizer panics THEN panic goes through",
			given: newControlPlane(t,
				mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
				mockWorkflowExecutionEngine{
					descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
					executionResults:  []proxias.WorkflowExecutionResult{{WorkflowID: "some_workflow"}},
				},
				proxias.WorkflowExecutionSynthesizerFunc(func(context.Context, proxias.WorkflowExecutionSynthesis) (proxias.ChatCompletionResponse, error) {
					panic(panicValue)
				}),
				mockReasoningEngine(someResponse, nil),
			),
		},
		{
			name: "GIVEN ReasoningEngine panics THEN panic goes through",
			given: newControlPlane(t,
				mockClassifier(proxias.Intent{}, nil),
				mockWorkflowExecutionEngine{descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor}},
				mockWorkflowExecutionSynthesizer(someResponse, nil),
				proxias.ReasoningEngineFunc(func(context.Context, proxias.ChatCompletionRequest, []proxias.WorkflowDescriptor) (proxias.ChatCompletionResponse, error) {
					panic(panicValue)
				}),
			),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.PanicsWithValue(t, panicValue, func() {
				_, _ = tt.given.Process(context.Background(), someRequest)
			})
		})
	}
}

type panicWorkflowExecutionEngine struct {
	panicMsg string
}

func (e *panicWorkflowExecutionEngine) WorkflowDescriptors(context.Context, proxias.ChatCompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return []proxias.WorkflowDescriptor{someWorkflowDescriptor}, nil
}

func (e *panicWorkflowExecutionEngine) ExecuteWorkflows(context.Context, proxias.ChatCompletionRequest, []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
	panic(e.panicMsg)
}

// newControlPlane makes a ControlPlane of the two middleware, which the components are given to.
func newControlPlane(tb testing.TB, classifier proxias.IntentClassifier, engine proxias.WorkflowExecutionEngine, synthesizer proxias.WorkflowExecutionSynthesizer, reasoner proxias.ReasoningEngine) proxias.ControlPlane {
	tb.Helper()
	interception, err := proxias.NewWorkflowInterceptionMiddleware(classifier, engine, synthesizer)
	require.NoError(tb, err)
	reasoning, err := proxias.NewReasoningMiddleware(reasoner)
	require.NoError(tb, err)
	controlPlane, err := proxias.NewControlPlane(interception, reasoning)
	require.NoError(tb, err)
	return controlPlane
}

func TestNewControlPlane(t *testing.T) {
	tests := []struct {
		name        string
		given       []proxias.Middleware
		errContains string
	}{
		{
			name:  "canonical",
			given: []proxias.Middleware{passThrough},
		},
		{
			name:        "GIVEN no middleware THEN it fails",
			errContains: "needs a Middleware",
		},
		{
			name:        "GIVEN a nil middleware THEN it fails and says which one",
			given:       []proxias.Middleware{passThrough, nil},
			errContains: "the one at 1 is nil",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			controlPlane, err := proxias.NewControlPlane(tt.given...)

			if tt.errContains != "" {
				assert.ErrorContains(t, err, tt.errContains)
				assert.Nil(t, controlPlane)
				return
			}
			require.NoError(t, err)
			assert.NotNil(t, controlPlane)
		})
	}
}

// passThrough is a middleware that leaves the request to the rest of the chain.
func passThrough(next proxias.ProcessFunc) proxias.ProcessFunc { return next }
