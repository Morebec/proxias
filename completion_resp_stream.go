package proxias

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// This file is the streamed side of the response, see completion_resp.go: a stream is Server-Sent Events whose data
// are ChatCompletionChunks, ended by [DONE].

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
	Index    int                             `json:"index"`
	ID       string                          `json:"id,omitempty"`
	Type     string                          `json:"type,omitempty"`
	Function ChatCompletionFunctionCallDelta `json:"function"`
}

// ChatCompletionFunctionCallDelta is a piece of the function of a tool call: its name in the first piece, then
// pieces of its arguments. Unlike ChatFunctionCall, an empty field is left out.
type ChatCompletionFunctionCallDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
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

// emitCompletion emits completion as the chunks of a stream: each choice in one piece, then its finish reason, and
// the usage last, if includeUsage.
func emitCompletion(emit func(ChatCompletionChunk) error, completion ChatCompletion, includeUsage bool) error {
	chunk := func(choices []ChatCompletionChunkChoice) ChatCompletionChunk {
		return ChatCompletionChunk{
			ID: completion.ID, Object: "chat.completion.chunk", Created: completion.Created, Model: completion.Model,
			Choices: choices,
		}
	}

	for _, choice := range completion.Choices {
		delta := ChatCompletionDelta{Role: choice.Message.Role, Content: choice.Message.Content.Text()}
		for i, call := range choice.Message.ToolCalls {
			delta.ToolCalls = append(delta.ToolCalls, ChatCompletionToolCallDelta{
				Index: i, ID: call.ID, Type: call.Type,
				Function: ChatCompletionFunctionCallDelta{Name: call.Function.Name, Arguments: call.Function.Arguments},
			})
		}
		finish := choice.FinishReason
		if err := emit(chunk([]ChatCompletionChunkChoice{{Index: choice.Index, Delta: delta}})); err != nil {
			return err
		}
		if err := emit(chunk([]ChatCompletionChunkChoice{{Index: choice.Index, FinishReason: &finish}})); err != nil {
			return err
		}
	}

	if includeUsage && completion.Usage != nil {
		last := chunk([]ChatCompletionChunkChoice{})
		last.Usage = completion.Usage
		return emit(last)
	}
	return nil
}
