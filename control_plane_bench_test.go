package proxias_test

import (
	"context"
	"testing"

	"github.com/morebec/proxias"
)

// BenchmarkControlPlaneProcess measures the ControlPlane's own overhead: the components answer instantly.
func BenchmarkControlPlaneProcess(b *testing.B) {
	controlPlane := proxias.NewControlPlane(
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
