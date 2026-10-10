package proxias

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// This file is the response side of the model of Proxias, see completion_req.go. The answer to a
// ChatCompletionRequest is a ChatCompletionResponse, which is the HTTP response (status, headers, body) that
// the host writes back. The body of a response is either a ChatCompletion, or a stream of
// ChatCompletionChunks: NewChatCompletionResponse and NewChatCompletionStream make a response of them. A
// response that comes from an upstream provider is forwarded as it is, without being decoded.

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

// NewChatCompletionResponse makes the response of a whole answer.
func NewChatCompletionResponse(completion ChatCompletion) (ChatCompletionResponse, error) {
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

// NewChatCompletionStream makes the response of a streamed answer, of the chunks that produce emits: they are
// events of the stream, followed by its end. The response is returned at once and produce runs in the
// background, until it returns or the response's Body is closed, which makes emit fail. If produce returns an
// error, reading the Body fails with it, and the stream has no end event.
func NewChatCompletionStream(produce func(emit func(ChatCompletionChunk) error) error) ChatCompletionResponse {
	reader, writer := io.Pipe()

	go func() {
		emit := func(chunk ChatCompletionChunk) error {
			data, err := json.Marshal(chunk)
			if err != nil {
				return err
			}
			_, err = fmt.Fprintf(writer, "data: %s\n\n", data)
			return err
		}
		if err := produce(emit); err != nil {
			_ = writer.CloseWithError(err)
			return
		}
		_, err := io.WriteString(writer, "data: [DONE]\n\n")
		_ = writer.CloseWithError(err) // nil closes with io.EOF
	}()

	return ChatCompletionResponse{
		Status: http.StatusOK,
		Header: http.Header{"Content-Type": {"text/event-stream"}, "Cache-Control": {"no-cache"}},
		Body:   reader,
	}
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

// ChatCompletionChunk is a piece of a streamed answer: the data of an event of the stream.
type ChatCompletionChunk struct {
	ID      string                      `json:"id"`
	Object  string                      `json:"object"`
	Created int64                       `json:"created"`
	Model   string                      `json:"model"`
	Choices []ChatCompletionChunkChoice `json:"choices"`
	// Usage is only in the last chunk, and only if the request asked for it.
	Usage *ChatUsage `json:"usage,omitempty"`
}

// ChatCompletionChunkChoice is the piece of one answer of a ChatCompletionChunk. FinishReason is nil until
// the last chunk of the answer, which is null on the wire.
type ChatCompletionChunkChoice struct {
	Index        int                 `json:"index"`
	Delta        ChatCompletionDelta `json:"delta"`
	FinishReason *ChatFinishReason   `json:"finish_reason"`
}

// ChatCompletionDelta is what a chunk adds to the answer.
type ChatCompletionDelta struct {
	Role      ChatMessageRole               `json:"role,omitempty"`
	Content   string                        `json:"content,omitempty"`
	ToolCalls []ChatCompletionToolCallDelta `json:"tool_calls,omitempty"`
}

// ChatCompletionToolCallDelta is a piece of a tool call: the first one gives the ID and the name of the
// function, the next ones pieces of its arguments. Index tells which call of the answer it is a piece of.
type ChatCompletionToolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function"`
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

// ChatFinishReason is why a model stopped: "stop", "length", "tool_calls" or "content_filter".
type ChatFinishReason string

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
