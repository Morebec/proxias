package proxias_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

var someRequest = proxias.ChatCompletionRequest{Model: "m"}
var someResponse = proxias.ChatCompletionResponse{Status: 200}

// passThrough is a middleware that leaves the request to the rest of the chain.
func passThrough(next proxias.ProcessFunc) proxias.ProcessFunc { return next }

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

func TestControlPlaneProcessRunsTheStepsInOrder(t *testing.T) {
	// Given two steps, and a last one that answers
	var order []string
	var answered proxias.ChatCompletionRequest
	controlPlane, err := proxias.NewControlPlane(
		recordingStep(&order, "first"),
		recordingStep(&order, "second"),
		answering(&answered),
	)
	require.NoError(t, err)

	// When
	got, err := controlPlane.Process(t.Context(), someRequest)

	// Then each step ran, in the order given, and the last one got the request and answered
	require.NoError(t, err)
	assert.Equal(t, []string{"first", "second"}, order)
	assert.Equal(t, someRequest, answered)
	assert.Equal(t, someResponse, got)
}

func TestControlPlaneProcessBypassesTheStepsAfterTheOneThatAnswers(t *testing.T) {
	// Given a step that answers, followed by one that must not run
	var order []string
	var answered proxias.ChatCompletionRequest
	controlPlane, err := proxias.NewControlPlane(answering(&answered), recordingStep(&order, "never"))
	require.NoError(t, err)

	// When
	_, err = controlPlane.Process(t.Context(), someRequest)

	// Then
	require.NoError(t, err)
	assert.Empty(t, order)
}

func TestControlPlaneProcessFailsWhenNoStepAnswers(t *testing.T) {
	// Given a chain whose steps all pass the request on
	controlPlane, err := proxias.NewControlPlane(passThrough)
	require.NoError(t, err)

	// When
	_, err = controlPlane.Process(t.Context(), someRequest)

	// Then
	assert.EqualError(t, err, "proxias: no middleware answered the request")
}

func TestControlPlaneProcessReturnsContextErrorBeforeProcessing(t *testing.T) {
	// Given a cancelled context, and a step that must not run
	var order []string
	controlPlane, err := proxias.NewControlPlane(recordingStep(&order, "never"), passThrough)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	// When
	_, err = controlPlane.Process(ctx, someRequest)

	// Then it is the context's error, and nothing ran
	assert.Equal(t, context.Canceled, err)
	assert.Empty(t, order)
}

func TestControlPlaneProcessPrefixesStepErrorsOnce(t *testing.T) {
	// Given a step that fails with an error of its own
	cause := errors.New("boom")
	failing := func(proxias.ProcessFunc) proxias.ProcessFunc {
		return func(context.Context, proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			return proxias.ChatCompletionResponse{}, fmt.Errorf("stage: %w", cause)
		}
	}
	controlPlane, err := proxias.NewControlPlane(failing)
	require.NoError(t, err)

	// When
	_, err = controlPlane.Process(t.Context(), someRequest)

	// Then the prefix is added once, and the cause is found
	assert.EqualError(t, err, "proxias: stage: boom")
	assert.ErrorIs(t, err, cause)
}

func TestControlPlaneProcessReturnsAStepsContextErrorAsIs(t *testing.T) {
	// Given a step that cancels the context and returns its error
	ctx, cancel := context.WithCancel(t.Context())
	cancelling := func(proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, _ proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			cancel()
			return proxias.ChatCompletionResponse{}, ctx.Err()
		}
	}
	controlPlane, err := proxias.NewControlPlane(cancelling)
	require.NoError(t, err)

	// When
	_, err = controlPlane.Process(ctx, someRequest)

	// Then it is the context's error, unwrapped
	assert.Equal(t, context.Canceled, err)
}

func TestControlPlaneProcessDoesNotRecoverPanics(t *testing.T) {
	// Given a step that panics
	panicking := func(proxias.ProcessFunc) proxias.ProcessFunc {
		return func(context.Context, proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			panic("boom")
		}
	}
	controlPlane, err := proxias.NewControlPlane(panicking)
	require.NoError(t, err)

	// When, Then the panic goes through, for the host to recover
	assert.PanicsWithValue(t, "boom", func() {
		_, _ = controlPlane.Process(t.Context(), someRequest)
	})
}
