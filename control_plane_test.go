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

func (e mockWorkflowExecutionEngine) WorkflowDescriptors(context.Context, proxias.CompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return e.descriptorResults, e.describeErr
}

func (e mockWorkflowExecutionEngine) ExecuteWorkflows(context.Context, proxias.CompletionRequest, []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
	return e.executionResults, e.executeErr
}

var someRequest = proxias.CompletionRequest{Model: "m"}
var someResponse = proxias.CompletionResponse{Status: 200}

func mockClassifier(intent proxias.Intent, err error) proxias.IntentClassifier {
	return proxias.IntentClassifierFunc(func(context.Context, proxias.CompletionRequest, []proxias.WorkflowDescriptor) (proxias.Intent, error) {
		return intent, err
	})
}

func mockWorkflowExecutionSynthesizer(resp proxias.CompletionResponse, err error) proxias.WorkflowExecutionSynthesizer {
	return proxias.WorkflowExecutionSynthesizerFunc(func(context.Context, proxias.WorkflowExecutionSynthesis) (proxias.CompletionResponse, error) {
		return resp, err
	})
}

func mockReasoningEngine(resp proxias.CompletionResponse, err error) proxias.ReasoningEngine {
	return proxias.ReasoningEngineFunc(func(context.Context, proxias.CompletionRequest, []proxias.WorkflowDescriptor) (proxias.CompletionResponse, error) {
		return resp, err
	})
}

func recordingWorkflowExecutionSynthesizer(recorded *proxias.WorkflowExecutionSynthesis) proxias.WorkflowExecutionSynthesizer {
	return proxias.WorkflowExecutionSynthesizerFunc(func(_ context.Context, synthesis proxias.WorkflowExecutionSynthesis) (proxias.CompletionResponse, error) {
		*recorded = synthesis
		return someResponse, nil
	})
}

func recordingReasoningEngine(recordedReq *proxias.CompletionRequest, recordedDesc *[]proxias.WorkflowDescriptor) proxias.ReasoningEngine {
	return proxias.ReasoningEngineFunc(func(_ context.Context, req proxias.CompletionRequest, desc []proxias.WorkflowDescriptor) (proxias.CompletionResponse, error) {
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
				synthesizer: mockWorkflowExecutionSynthesizer(proxias.CompletionResponse{}, assert.AnError),
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
				reasoningEngine: mockReasoningEngine(proxias.CompletionResponse{}, assert.AnError),
			},
			then: expect{
				errContains: "reason",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var recordedReq proxias.CompletionRequest
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
		proxias.WorkflowExecutionSynthesizerFunc(func(context.Context, proxias.WorkflowExecutionSynthesis) (proxias.CompletionResponse, error) {
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

func (e *delayedWorkflowExecutionEngine) WorkflowDescriptors(context.Context, proxias.CompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return e.descriptorResults, nil
}

func (e *delayedWorkflowExecutionEngine) ExecuteWorkflows(ctx context.Context, _ proxias.CompletionRequest, _ []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
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
				proxias.IntentClassifierFunc(func(context.Context, proxias.CompletionRequest, []proxias.WorkflowDescriptor) (proxias.Intent, error) {
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
				proxias.WorkflowExecutionSynthesizerFunc(func(context.Context, proxias.WorkflowExecutionSynthesis) (proxias.CompletionResponse, error) {
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
				proxias.ReasoningEngineFunc(func(context.Context, proxias.CompletionRequest, []proxias.WorkflowDescriptor) (proxias.CompletionResponse, error) {
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

func (e *panicWorkflowExecutionEngine) WorkflowDescriptors(context.Context, proxias.CompletionRequest) ([]proxias.WorkflowDescriptor, error) {
	return []proxias.WorkflowDescriptor{someWorkflowDescriptor}, nil
}

func (e *panicWorkflowExecutionEngine) ExecuteWorkflows(context.Context, proxias.CompletionRequest, []proxias.WorkflowExecutionIntent) ([]proxias.WorkflowExecutionResult, error) {
	panic(e.panicMsg)
}

// newControlPlane makes a ControlPlane of components that are all given.
func newControlPlane(tb testing.TB, classifier proxias.IntentClassifier, engine proxias.WorkflowExecutionEngine, synthesizer proxias.WorkflowExecutionSynthesizer, reasoner proxias.ReasoningEngine) proxias.ControlPlane {
	tb.Helper()
	controlPlane, err := proxias.NewControlPlane(classifier, engine, synthesizer, reasoner)
	require.NoError(tb, err)
	return controlPlane
}

func TestNewControlPlane(t *testing.T) {
	type assumptions struct {
		classifier              proxias.IntentClassifier
		workflowExecutionEngine proxias.WorkflowExecutionEngine
		synthesizer             proxias.WorkflowExecutionSynthesizer
		reasoningEngine         proxias.ReasoningEngine
	}
	tests := []struct {
		name  string
		given assumptions
		then  string // error message if any
	}{
		{
			name: "GIVEN all four components THEN a control plane is created",
			given: assumptions{
				classifier:              mockClassifier(proxias.Intent{}, nil),
				workflowExecutionEngine: mockWorkflowExecutionEngine{},
				synthesizer:             mockWorkflowExecutionSynthesizer(someResponse, nil),
				reasoningEngine:         mockReasoningEngine(someResponse, nil),
			},
			then: "",
		},
		{
			name: "GIVEN no IntentClassifier THEN it fails",
			given: assumptions{
				classifier:              nil,
				workflowExecutionEngine: mockWorkflowExecutionEngine{},
				synthesizer:             mockWorkflowExecutionSynthesizer(someResponse, nil),
				reasoningEngine:         mockReasoningEngine(someResponse, nil),
			},
			then: "IntentClassifier",
		},
		{
			name: "GIVEN no WorkflowExecutionEngine THEN it fails",
			given: assumptions{
				classifier:              mockClassifier(proxias.Intent{}, nil),
				workflowExecutionEngine: nil,
				synthesizer:             mockWorkflowExecutionSynthesizer(someResponse, nil),
				reasoningEngine:         mockReasoningEngine(someResponse, nil),
			},
			then: "WorkflowExecutionEngine",
		},
		{
			name: "GIVEN no WorkflowExecutionSynthesizer THEN it fails",
			given: assumptions{
				classifier:              mockClassifier(proxias.Intent{}, nil),
				workflowExecutionEngine: mockWorkflowExecutionEngine{},
				synthesizer:             nil,
				reasoningEngine:         mockReasoningEngine(someResponse, nil),
			},
			then: "Synthesizer",
		},
		{
			name: "GIVEN no ReasoningEngine THEN it fails",
			given: assumptions{
				classifier:              mockClassifier(proxias.Intent{}, nil),
				workflowExecutionEngine: mockWorkflowExecutionEngine{},
				synthesizer:             mockWorkflowExecutionSynthesizer(someResponse, nil),
				reasoningEngine:         nil,
			},
			then: "ReasoningEngine",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			controlPlane, err := proxias.NewControlPlane(tt.given.classifier, tt.given.workflowExecutionEngine, tt.given.synthesizer, tt.given.reasoningEngine)

			if tt.then == "" {
				require.NoError(t, err)
				assert.NotNil(t, controlPlane)
			} else {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.then)
				assert.Nil(t, controlPlane)
			}
		})
	}
}
