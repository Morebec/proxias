package proxias

import (
	"context"
	"fmt"
)

// NewWorkflowInterceptionMiddleware makes the middleware that runs the workflows a request asks for: the
// classifier names them, the engine runs them, and the synthesizer makes the response. A request that asks for
// no workflow is passed on to the rest of the chain, with the descriptors of the workflows in its context. It
// returns an error if one of the components is nil.
func NewWorkflowInterceptionMiddleware(classifier IntentClassifier, engine WorkflowExecutionEngine, synthesizer WorkflowExecutionSynthesizer) (Middleware, error) {
	switch {
	case classifier == nil:
		return nil, fmt.Errorf("proxias: NewWorkflowInterceptionMiddleware needs an IntentClassifier")
	case engine == nil:
		return nil, fmt.Errorf("proxias: NewWorkflowInterceptionMiddleware needs a WorkflowExecutionEngine")
	case synthesizer == nil:
		return nil, fmt.Errorf("proxias: NewWorkflowInterceptionMiddleware needs a WorkflowExecutionSynthesizer")
	}

	return func(next ProcessFunc) ProcessFunc {
		return func(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
			workflowDescs, err := engine.WorkflowDescriptors(ctx, req)
			if err != nil {
				return ChatCompletionResponse{}, fmt.Errorf("proxias: list workflow descriptors: %w", err)
			}

			// Without workflows there is nothing to choose amongst: no need to ask the classifier.
			var intent Intent
			if len(workflowDescs) != 0 {
				intent, err = classifier.Classify(ctx, req, workflowDescs)
				if err != nil {
					return ChatCompletionResponse{}, fmt.Errorf("proxias: classify intent: %w", err)
				}
			}

			if intent.IsReasoningIntent() {
				return next(WithWorkflowDescriptors(ctx, workflowDescs), req)
			}

			return executeAndSynthesize(ctx, req, intent, engine, synthesizer)
		}
	}, nil
}

// executeAndSynthesize runs the workflows of an intent, and synthesizes their results into the response.
func executeAndSynthesize(ctx context.Context, req ChatCompletionRequest, intent Intent, engine WorkflowExecutionEngine, synthesizer WorkflowExecutionSynthesizer) (ChatCompletionResponse, error) {
	results, err := engine.ExecuteWorkflows(ctx, req, intent.WorkflowExecutions)
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

	response, err := synthesizer.Synthesize(ctx, WorkflowExecutionSynthesis{
		Request:          req,
		Intent:           intent,
		ExecutionResults: results,
	})
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("proxias: synthesize: %w", err)
	}

	return response, nil
}
