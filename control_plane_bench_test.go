package proxias_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

// newControlPlane makes the usual ControlPlane, of the two middleware that the components are given to.
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

// BenchmarkControlPlaneProcess measures the ControlPlane's own overhead: the components answer instantly.
func BenchmarkControlPlaneProcess(b *testing.B) {
	controlPlane := newControlPlane(b,
		mockClassifier(proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{someWorkflowExecutionIntent}}, nil),
		mockWorkflowExecutionEngine{
			descriptorResults: []proxias.WorkflowDescriptor{someWorkflowDescriptor},
			executionResults:  []proxias.WorkflowExecutionResult{{WorkflowID: "some_workflow", Status: "completed"}},
		},
		mockWorkflowExecutionSynthesizer(someResponse, nil),
		mockReasoningEngine(someResponse, nil),
	)
	ctx := context.Background()

	b.ReportAllocs()
	for b.Loop() {
		if _, err := controlPlane.Process(ctx, someRequest); err != nil {
			b.Fatal(err)
		}
	}
}
