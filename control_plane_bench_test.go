package proxias_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

// newControlPlane makes the usual ControlPlane, of the two middleware that the components are given to.
func newControlPlane(tb testing.TB, classifier proxias.IntentClassifier, engine proxias.ActionExecutionEngine, synthesizer proxias.ActionExecutionSynthesizer, reasoner proxias.ReasoningEngine) proxias.ControlPlane {
	tb.Helper()
	interception, err := proxias.NewActionInterceptionMiddleware(classifier, engine, synthesizer)
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
		mockClassifier(proxias.Intent{ActionExecutions: []proxias.ActionExecutionIntent{someActionExecutionIntent}}, nil),
		mockActionExecutionEngine{
			descriptorResults: []proxias.ActionDescriptor{someActionDescriptor},
			executionResults:  []proxias.ActionExecutionResult{{ActionID: "some_action", Status: "completed"}},
		},
		mockActionExecutionSynthesizer(someResponse, nil),
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
