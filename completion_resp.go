package proxias

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const (
	// ChatFinishReasonStop is a natural end, or a stop sequence.
	ChatFinishReasonStop ChatFinishReason = "stop"
	// ChatFinishReasonLength is an answer cut at the limit of tokens.
	ChatFinishReasonLength ChatFinishReason = "length"
	// ChatFinishReasonToolCalls is an answer that asks for tool calls.
	ChatFinishReasonToolCalls ChatFinishReason = "tool_calls"
	// ChatFinishReasonContentFilter is an answer the provider declined to give.
	ChatFinishReasonContentFilter ChatFinishReason = "content_filter"
)

// ChatFinishReason is why a model stopped: "stop", "length", "tool_calls" or "content_filter".
type ChatFinishReason string

// This file is the response side of the model of Proxias, see completion_req.go. The answer to a
// ChatCompletionRequest is a ChatCompletionResponse, which is the HTTP response (status, headers, body) that
// the host writes back. The body of a response is either a ChatCompletion, or a stream of
// ChatCompletionChunks. NewChatCompletionResponse makes the response of a ChatCompletion, streamed or not as the
// request asks; NewChatCompletionStream makes a stream of chunks produced over time. A response that comes from an
// upstream provider is forwarded as it is, without being decoded.

// ChatCompletionResponse is what the ControlPlane answers: an HTTP response that the host writes back to its
// client, a stream included. It is the answer of OpenAI's chat completions API, forwarded as it came from
// the model's provider or made with NewChatCompletionResponse and NewChatCompletionStream.
//
// Whoever receives a ChatCompletionResponse owns its Body, if there is one, and must close it. A response that
// comes with an error has none.
type ChatCompletionResponse struct {
	// Status is the HTTP status code of the response.
	Status int
	// Header is the headers of the response: its Content-Type, "text/event-stream" for a stream...
	Header http.Header
	// Body is the content of the response, if any.
	Body io.ReadCloser
}

// NewChatCompletionResponse makes the response of completion to req: the completion as it is or, if req asked for
// a stream, a stream of it, its usage at the end if req asked for it. Make completion with ChatCompletionBuilder.
// Chunks produced over time, as a provider streams them, are for NewChatCompletionStream.
func NewChatCompletionResponse(req ChatCompletionRequest, completion ChatCompletion) (ChatCompletionResponse, error) {
	if req.Stream {
		return NewChatCompletionStream(func(emit func(ChatCompletionChunk) error) error {
			return emitCompletion(emit, completion, req.UsageRequested())
		}), nil
	}

	body, err := json.Marshal(completion)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("proxias: encode the chat completion: %w", err)
	}
	return ChatCompletionResponse{
		Status: http.StatusOK,
		Header: http.Header{"Content-Type": {"application/json"}},
		Body:   io.NopCloser(bytes.NewReader(body)),
	}, nil
}

// ChatCompletion is a whole answer: the body of a response to a request that did not ask for a stream.
type ChatCompletion struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   *ChatUsage             `json:"usage,omitempty"`
}

// ChatCompletionChoice is one answer of a ChatCompletion.
type ChatCompletionChoice struct {
	Index        int              `json:"index"`
	Message      ChatMessage      `json:"message"`
	FinishReason ChatFinishReason `json:"finish_reason"`
}

// ChatUsage is what an answer cost, in tokens.
type ChatUsage struct {
	PromptTokens        int64                    `json:"prompt_tokens"`
	CompletionTokens    int64                    `json:"completion_tokens"`
	TotalTokens         int64                    `json:"total_tokens"`
	PromptTokensDetails *ChatPromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

// ChatPromptTokensDetails tells apart the prompt tokens that were read from the provider's cache.
type ChatPromptTokensDetails struct {
	CachedTokens int64 `json:"cached_tokens"`
}
