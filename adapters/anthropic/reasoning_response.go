package anthropic

import (
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"

	"github.com/morebec/proxias"
)

// mapFinishReason maps why Claude stopped.
func mapFinishReason(stopReason anthropicsdk.StopReason) proxias.ChatFinishReason {
	switch stopReason {
	case anthropicsdk.StopReasonToolUse:
		return proxias.ChatFinishReasonToolCalls
	case anthropicsdk.StopReasonMaxTokens:
		return proxias.ChatFinishReasonLength
	case anthropicsdk.StopReasonRefusal:
		return proxias.ChatFinishReasonContentFilter
	}
	return proxias.ChatFinishReasonStop
}

// mapChatUsage maps what the answer cost. The prompt counts every input token, the ones read from or written to
// the cache included.
func mapChatUsage(u anthropicsdk.Usage, outputTokens int64) *proxias.ChatUsage {
	prompt := u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	return &proxias.ChatUsage{
		PromptTokens:        prompt,
		CompletionTokens:    outputTokens,
		TotalTokens:         prompt + outputTokens,
		PromptTokensDetails: &proxias.ChatPromptTokensDetails{CachedTokens: u.CacheReadInputTokens},
	}
}

// mapChatCompletion maps a whole answer of Claude. Thinking is left out: chat completions have no place for it.
func mapChatCompletion(message *anthropicsdk.Message) proxias.ChatCompletion {
	answer := proxias.ChatMessage{Role: proxias.ChatMessageRoleAssistant}

	var text string
	for _, block := range message.Content {
		switch b := block.AsAny().(type) {
		case anthropicsdk.TextBlock:
			text += b.Text
		case anthropicsdk.ToolUseBlock:
			answer.ToolCalls = append(answer.ToolCalls, proxias.ChatToolCall{
				ID:       b.ID,
				Type:     "function",
				Function: proxias.ChatFunctionCall{Name: b.Name, Arguments: string(b.Input)},
			})
		}
	}
	if text != "" {
		answer.Content = proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: text}}
	}

	return proxias.ChatCompletion{
		ID:      message.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   message.Model,
		Choices: []proxias.ChatCompletionChoice{{Message: answer, FinishReason: mapFinishReason(message.StopReason)}},
		Usage:   mapChatUsage(message.Usage, message.Usage.OutputTokens),
	}
}

// mapChunks turns the events of a Claude stream into chat completion mapChunks, emitted as they come. The stream
// has its first event read already, which is the current one. It ends with the stream, and closes it.
func mapChunks(stream *ssestream.Stream[anthropicsdk.MessageStreamEventUnion], includeUsage bool) func(emit func(proxias.ChatCompletionChunk) error) error {
	return func(emit func(proxias.ChatCompletionChunk) error) error {
		defer func() { _ = stream.Close() }()

		var (
			id, model    string
			created      = time.Now().Unix()
			inputUsage   anthropicsdk.Usage
			outputTokens int64
			stopReason   anthropicsdk.StopReason
			// toolCalls maps the index of a tool_use content block to its index among the tool calls.
			toolCalls = map[int64]int{}
		)
		chunk := func(delta proxias.ChatCompletionDelta, finish *proxias.ChatFinishReason) error {
			return emit(proxias.ChatCompletionChunk{
				ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
				Choices: []proxias.ChatCompletionChunkChoice{{Delta: delta, FinishReason: finish}},
			})
		}

		for event := stream.Current(); ; event = stream.Current() {
			var err error
			switch event.Type {
			case "message_start":
				id, model, inputUsage = event.Message.ID, event.Message.Model, event.Message.Usage
				err = chunk(proxias.ChatCompletionDelta{Role: proxias.ChatMessageRoleAssistant}, nil)

			case "content_block_start":
				if event.ContentBlock.Type == "tool_use" {
					index := len(toolCalls)
					toolCalls[event.Index] = index
					call := proxias.ChatCompletionToolCallDelta{Index: index, ID: event.ContentBlock.ID, Type: "function"}
					call.Function.Name = event.ContentBlock.Name
					err = chunk(proxias.ChatCompletionDelta{ToolCalls: []proxias.ChatCompletionToolCallDelta{call}}, nil)
				}

			case "content_block_delta":
				switch event.Delta.Type {
				case "text_delta":
					err = chunk(proxias.ChatCompletionDelta{Content: event.Delta.Text}, nil)
				case "input_json_delta":
					if index, ok := toolCalls[event.Index]; ok && event.Delta.PartialJSON != "" {
						call := proxias.ChatCompletionToolCallDelta{Index: index}
						call.Function.Arguments = event.Delta.PartialJSON
						err = chunk(proxias.ChatCompletionDelta{ToolCalls: []proxias.ChatCompletionToolCallDelta{call}}, nil)
					}
				}

			case "message_delta":
				stopReason = event.Delta.StopReason
				outputTokens = event.Usage.OutputTokens

			case "message_stop":
				reason := mapFinishReason(stopReason)
				if err := chunk(proxias.ChatCompletionDelta{}, &reason); err != nil {
					return err
				}
				if includeUsage {
					return emit(proxias.ChatCompletionChunk{
						ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
						Choices: []proxias.ChatCompletionChunkChoice{},
						Usage:   mapChatUsage(inputUsage, outputTokens),
					})
				}
				return nil
			}
			if err != nil {
				return err // the harness went away
			}
			if !stream.Next() {
				return stream.Err()
			}
		}
	}
}
