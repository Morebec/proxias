package proxias

import (
	"context"
	"errors"
	"fmt"
)

// ControlPlane turns a request into a response, either by running workflows or by reasoning.
// It is the entry point of the SDK: an IntentClassifier names which of the
// WorkflowExecutionEngine's workflows the request asks for (and with which parameters), the
// WorkflowExecutionEngine runs them, and the WorkflowExecutionSynthesizer makes the response. A request that
// asks for no workflow is answered by the ReasoningEngine.
type ControlPlane interface {
	// Process answers req. An error is an infrastructure failure of one of the stages, or of the context:
	// a workflow that cannot run for a reason the user should hear of is not an error, it is a
	// WorkflowExecutionResult for the synthesizer.
	//
	// Process does not recover from panics: the host does, if it wants to.
	Process(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error)
}

// controlPlane is the ControlPlane made by NewControlPlane. It is safe for concurrent use as long as the
// components it was made of are.
type controlPlane struct {
	intentClassifier             IntentClassifier
	workflowExecutionEngine      WorkflowExecutionEngine
	workflowExecutionSynthesizer WorkflowExecutionSynthesizer
	reasoningEngine              ReasoningEngine
}

// NewControlPlane makes a ControlPlane. It returns an error if any of the four components is nil.
func NewControlPlane(
	classifier IntentClassifier,
	workflowEngine WorkflowExecutionEngine,
	synthesizer WorkflowExecutionSynthesizer,
	reasoningEngine ReasoningEngine,
	options ...ControlPlaneOption,
) (ControlPlane, error) {
	switch {
	case classifier == nil:
		return nil, errors.New("proxias: NewControlPlane needs an IntentClassifier")
	case workflowEngine == nil:
		return nil, errors.New("proxias: NewControlPlane needs a WorkflowExecutionEngine")
	case synthesizer == nil:
		return nil, errors.New("proxias: NewControlPlane needs a Synthesizer")
	case reasoningEngine == nil:
		return nil, errors.New("proxias: NewControlPlane needs a ReasoningEngine")
	}

	e := &controlPlane{}
	for _, option := range options {
		option(e)
	}

	e.intentClassifier = classifier
	e.workflowExecutionEngine = workflowEngine
	e.workflowExecutionSynthesizer = synthesizer
	e.reasoningEngine = reasoningEngine

	return e, nil
}

// Process implements ControlPlane.
func (e *controlPlane) Process(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	if err := ctx.Err(); err != nil {
		return ChatCompletionResponse{}, err
	}

	workflowDescs, err := e.workflowExecutionEngine.WorkflowDescriptors(ctx, req)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("proxias: list workflow descriptors: %w", err)
	}

	// Without workflows there is nothing to choose amongst: no need to ask the classifier.
	var intent Intent

	if len(workflowDescs) != 0 {
		intent, err = e.intentClassifier.Classify(ctx, req, workflowDescs)
		if err != nil {
			return ChatCompletionResponse{}, fmt.Errorf("proxias: classify intent: %w", err)
		}
	}

	if intent.IsReasoningIntent() {
		return e.processOpenEndedReasoningIntent(ctx, req, workflowDescs)
	}

	return e.processWorkflowExecutionIntent(ctx, req, intent)
}

func (e *controlPlane) processOpenEndedReasoningIntent(
	ctx context.Context,
	req ChatCompletionRequest,
	workflowDescs []WorkflowDescriptor,
) (ChatCompletionResponse, error) {
	response, err := e.reasoningEngine.Reason(ctx, req, workflowDescs)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("proxias: reason: %w", err)
	}

	return response, nil
}

func (e *controlPlane) processWorkflowExecutionIntent(
	ctx context.Context,
	req ChatCompletionRequest,
	intent Intent,
) (ChatCompletionResponse, error) {
	results, err := e.workflowExecutionEngine.ExecuteWorkflows(ctx, req, intent.WorkflowExecutions)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("proxias: execute workflows: %w", err)
	}

	if len(results) != len(intent.WorkflowExecutions) {
		return ChatCompletionResponse{}, fmt.Errorf("proxias: execute workflows: got %d results for %d executions", len(results), len(intent.WorkflowExecutions))
	}

	// The workflows may have run past the end of the request: nobody is waiting for a response anymore.
	if err := ctx.Err(); err != nil {
		return ChatCompletionResponse{}, err
	}

	response, err := e.workflowExecutionSynthesizer.Synthesize(ctx, WorkflowExecutionSynthesis{
		Request:          req,
		Intent:           intent,
		ExecutionResults: results,
	})
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("proxias: synthesize: %w", err)
	}

	return response, nil
}
