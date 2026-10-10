package proxias_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

// answering is the last step of a chain: it records the request it receives, and answers.
func answering(recorded *proxias.ChatCompletionRequest) proxias.Middleware {
	return func(proxias.ProcessFunc) proxias.ProcessFunc {
		return func(_ context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			*recorded = req
			return someResponse, nil
		}
	}
}

// recordingStep is a middleware that records its name in order, and passes the request on.
func recordingStep(order *[]string, name string) proxias.Middleware {
	return func(next proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			*order = append(*order, name)
			return next(ctx, req)
		}
	}
}

func TestNewControlPlaneRunsTheStepsInOrder(t *testing.T) {
	// Given three steps, and a last one that answers
	var order []string
	var answered proxias.ChatCompletionRequest
	controlPlane, err := proxias.NewControlPlane(
		recordingStep(&order, "first"),
		recordingStep(&order, "second"),
		answering(&answered),
	)
	require.NoError(t, err)

	// When a request is processed
	_, err = controlPlane.Process(t.Context(), someRequest)

	// Then each step ran, in the order given, and the last one got the request
	require.NoError(t, err)
	assert.Equal(t, []string{"first", "second"}, order)
	assert.Equal(t, someRequest, answered)
}

func TestNewControlPlaneBypassesTheStepsAfterTheOneThatAnswers(t *testing.T) {
	// Given a step that answers, followed by one that must never run
	var order []string
	answerer := func(proxias.ProcessFunc) proxias.ProcessFunc {
		return func(context.Context, proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			return someResponse, nil
		}
	}
	controlPlane, err := proxias.NewControlPlane(answerer, recordingStep(&order, "never"))
	require.NoError(t, err)

	// When a request is processed
	got, err := controlPlane.Process(t.Context(), someRequest)

	// Then the answer is the one of the first step, and the rest did not run
	require.NoError(t, err)
	assert.Equal(t, someResponse, got)
	assert.Empty(t, order)
}

func TestNewControlPlaneFailsWhenNoStepAnswers(t *testing.T) {
	// Given a chain whose steps all pass the request on
	controlPlane, err := proxias.NewControlPlane(passThrough)
	require.NoError(t, err)

	// When a request is processed
	_, err = controlPlane.Process(t.Context(), someRequest)

	// Then it is an error
	assert.ErrorContains(t, err, "no middleware answered the request")
}

func TestNewControlPlaneChecksTheContextBeforeAnyStep(t *testing.T) {
	// Given a cancelled context, and a step that must not run
	var order []string
	controlPlane, err := proxias.NewControlPlane(recordingStep(&order, "never"), passThrough)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	// When a request is processed
	_, err = controlPlane.Process(ctx, someRequest)

	// Then it is the context's error, and nothing ran
	assert.ErrorIs(t, err, context.Canceled)
	assert.Empty(t, order)
}

func TestMiddlewareCanAddDataToTheContextForTheRestOfTheChain(t *testing.T) {
	// Given a step that puts the workflow descriptors in the context, before a step that reads them
	descs := []proxias.WorkflowDescriptor{someWorkflowDescriptor}
	setter := func(next proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			return next(proxias.WithWorkflowDescriptors(ctx, descs), req)
		}
	}
	var got []proxias.WorkflowDescriptor
	reader := func(proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, _ proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			got = proxias.WorkflowDescriptorsFrom(ctx)
			return someResponse, nil
		}
	}
	controlPlane, err := proxias.NewControlPlane(setter, reader)
	require.NoError(t, err)

	// When a request is processed
	_, err = controlPlane.Process(t.Context(), someRequest)

	// Then the reader sees the descriptors, and a context without them has none
	require.NoError(t, err)
	assert.Equal(t, descs, got)
	assert.Nil(t, proxias.WorkflowDescriptorsFrom(t.Context()))
}

// modelRouting is a middleware that picks the model from the conversation, as a host would route a request: a
// long conversation goes to the large model. It changes its own copy of the request, as a middleware must.
func modelRouting(next proxias.ProcessFunc) proxias.ProcessFunc {
	return func(ctx context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
		if len(req.Messages) > 2 {
			req.Model = "large"
		}
		return next(ctx, req)
	}
}

func TestMiddlewareRoutesTheModelAndTheRestOfTheRequestIsKept(t *testing.T) {
	// Given a request in which the harness set fields that the model does not name, and a routing step
	body := `{"model":"small","temperature":0.2,"stream_options":{"include_usage":true},"messages":[
		{"role":"system","content":"be brief"},
		{"role":"user","name":"ada","content":"one"},
		{"role":"assistant","content":"two"},
		{"role":"user","content":"three"}]}`
	var req proxias.ChatCompletionRequest
	require.NoError(t, json.Unmarshal([]byte(body), &req))
	var reached proxias.ChatCompletionRequest
	controlPlane, err := proxias.NewControlPlane(modelRouting, answering(&reached))
	require.NoError(t, err)

	// When it is processed
	_, err = controlPlane.Process(t.Context(), req)
	require.NoError(t, err)

	// Then the request that the last step received has the routed model, and everything else the harness sent
	assert.Equal(t, "large", reached.Model)
	routed, err := json.Marshal(reached)
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"large","temperature":0.2,"stream_options":{"include_usage":true},"messages":[
		{"role":"system","content":"be brief"},
		{"role":"user","name":"ada","content":"one"},
		{"role":"assistant","content":"two"},
		{"role":"user","content":"three"}]}`, string(routed))

	// And the caller's request is untouched: a middleware changes its copy
	assert.Equal(t, "small", req.Model)
}

func TestMiddlewareReplacesTheMessagesWithoutEditingTheCallersSlice(t *testing.T) {
	// Given a step that prepends a system message, by replacing the slice rather than editing it
	injection := func(next proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			system := proxias.ChatMessage{Role: proxias.ChatMessageRoleSystem, Content: proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: "be brief"}}}
			req.Messages = append([]proxias.ChatMessage{system}, req.Messages...)
			return next(ctx, req)
		}
	}
	original := []proxias.ChatMessage{{Role: proxias.ChatMessageRoleUser}}
	req := proxias.ChatCompletionRequest{Messages: original}
	var reached proxias.ChatCompletionRequest
	controlPlane, err := proxias.NewControlPlane(injection, answering(&reached))
	require.NoError(t, err)

	// When it is processed
	_, err = controlPlane.Process(t.Context(), req)

	// Then the step got two messages, and the caller still has one
	require.NoError(t, err)
	assert.Len(t, reached.Messages, 2)
	assert.Len(t, req.Messages, 1)
	assert.Equal(t, original, req.Messages)
}
