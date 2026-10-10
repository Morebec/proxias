// Package anthropic implements the components of Proxias with Claude, through the official Anthropic SDK.
//
// The host builds the SDK client, and so owns the API key, the timeout, the retries and the base URL, and
// always chooses the model. A failing call returns the SDK's error, wrapped: errors.As finds an
// *anthropic.Error of the SDK, and its status code.
package anthropic

import (
	"context"
	"errors"
	"fmt"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/morebec/proxias"
)

// ReasoningEngine is a proxias.ReasoningEngine that has Claude answer the conversation as it is. The
// request and the answer are those of the OpenAI chat completions API, which Claude's Messages API is mapped
// to and from: text, images, tool calls and results, cache breakpoints, and streaming.
type ReasoningEngine struct {
	claudeHelper
	MaxTokens int
}

type ReasoningEngineOption func(*ReasoningEngine)

// WithMaxTokens sets the maximum number of tokens for the ReasoningEngine.
func WithMaxTokens(maxTokens int) ReasoningEngineOption {
	return func(e *ReasoningEngine) {
		e.MaxTokens = maxTokens
	}
}

// NewReasoningEngine makes a ReasoningEngine that asks model through client.
func NewReasoningEngine(client anthropicsdk.Client, model anthropicsdk.Model, opts ...ReasoningEngineOption) *ReasoningEngine {
	e := &ReasoningEngine{claudeHelper: claudeHelper{client, model}}
	for _, o := range opts {
		o(e)
	}

	return e
}

// Reason implements proxias.ReasoningEngine. The workflow descriptors are not used.
//
// The answer is streamed if req.Stream is set. An error of the API is returned, wrapped, even for a stream:
// the first event is read before the response is.
func (e *ReasoningEngine) Reason(ctx context.Context, req proxias.ChatCompletionRequest, _ []proxias.WorkflowDescriptor) (proxias.ChatCompletionResponse, error) {
	params, err := mapChatCompletionRequestToMessageParams(req, e.model)
	if err != nil {
		return proxias.ChatCompletionResponse{}, fmt.Errorf("anthropic: map the request: %w", err)
	}

	if !req.Stream {
		message, err := e.client.Messages.New(ctx, params)
		if err != nil {
			return proxias.ChatCompletionResponse{}, fmt.Errorf("anthropic: %w", err)
		}
		return proxias.NewChatCompletionResponse(mapChatCompletion(message))
	}

	stream := e.client.Messages.NewStreaming(ctx, params)
	if !stream.Next() {
		err := stream.Err()
		_ = stream.Close()
		if err == nil {
			err = errors.New("the API answered nothing")
		}
		return proxias.ChatCompletionResponse{}, fmt.Errorf("anthropic: %w", err)
	}
	return proxias.NewChatCompletionStream(mapChunks(stream, req.StreamOptions.IncludeUsage)), nil
}
