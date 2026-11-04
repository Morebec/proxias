// Package openai is an OpenAI model for proxias, through the official SDK and
// its Chat Completions API: an proxias.LLM (NewLLM). The host builds the
// client (its key, timeout, retries), and chooses the model.
package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"

	"github.com/morebec/proxias"
)

// DefaultMaxTokens is the tokens an answer may take.
const DefaultMaxTokens = 4096

// LLM is an OpenAI model as an proxias.LLM. Make it with NewLLM.
type LLM struct {
	client    openai.Client
	model     string
	maxTokens int64
	// Streaming has Complete stream the answer: its text is handed out as the
	// model writes it, its tool calls once the answer is whole (a reasoning
	// the server streams is not handed out). Unset, the answer is read whole,
	// then handed out.
	Streaming bool
}

var _ proxias.LLM = LLM{}

// NewLLM returns the model of the client as an proxias.LLM: the system prompt
// is system messages, the conversation is the API's (tool calls in the
// assistant's messages, their results as tool messages), and the answer is read
// whole, then handed out as chunks (Streaming streams it). A model's thinking (another vendor's) is
// not given to it: the API takes none.
func NewLLM(client openai.Client, model string) LLM {
	return LLM{client: client, model: model, maxTokens: DefaultMaxTokens}
}

// Complete asks the model, and hands out its answer: its text, and its tool
// calls. The SDK's errors are returned as
// they are (errors.As an *openai.Error).
func (m LLM) Complete(ctx context.Context, req proxias.LLMRequest) iter.Seq2[proxias.LLMChunk, error] {
	return func(yield func(proxias.LLMChunk, error) bool) {
		params := openai.ChatCompletionNewParams{
			Model:               m.model,
			MaxCompletionTokens: openai.Int(m.maxTokens),
			Messages:            m.messages(req.Messages),
		}
		for _, tool := range req.Tools {
			params.Tools = append(
				params.Tools,
				openai.ChatCompletionFunctionTool(
					shared.FunctionDefinitionParam{
						Name:        tool.ID,
						Description: openai.String(tool.Description),
						Parameters:  m.parameters(tool.InputSchema),
					},
				),
			)
		}

		if m.Streaming {
			m.stream(ctx, params, yield)
			return
		}
		m.complete(ctx, params, yield)
	}
}

// complete reads the model's answer whole, then hands it out: its reasoning,
// its text, its tool calls.
func (m LLM) complete(
	ctx context.Context,
	params openai.ChatCompletionNewParams,
	yield func(proxias.LLMChunk, error) bool,
) {
	completion, err := m.client.Chat.Completions.New(ctx, params)
	if err != nil {
		yield(proxias.LLMChunk{}, err)
		return
	}
	if len(completion.Choices) == 0 {
		return
	}
	choice := completion.Choices[0]
	message := choice.Message

	// An OpenAI-compatible server (DeepSeek, vLLM, Ollama) may send the
	// model's reasoning: it is its thinking, given before its answer.
	reasoning, ok := message.JSON.ExtraFields["reasoning_content"]
	if ok {
		var thinking string
		err := json.Unmarshal([]byte(reasoning.Raw()), &thinking)
		if err == nil && thinking != "" {
			if !yield(proxias.LLMChunk{Thinking: thinking}, nil) {
				return
			}
		}
	}

	if message.Content != "" &&
		!yield(proxias.LLMChunk{Text: message.Content}, nil) {
		return
	}
	if !m.yieldToolCalls(message, yield) {
		return
	}
	stop := m.mapStopReason(choice.FinishReason)
	if stop != "" {
		yield(proxias.LLMChunk{StopReason: stop}, nil)
	}
}

// stream reads the model's answer as it comes: each piece of its text as it
// is written, its tool calls once the answer is whole.
func (m LLM) stream(
	ctx context.Context,
	params openai.ChatCompletionNewParams,
	yield func(proxias.LLMChunk, error) bool,
) {
	// An OpenAI-compatible server tells the usage of a stream only when it is
	// asked to.
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}
	stream := m.client.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()

	var answer openai.ChatCompletionAccumulator
	for stream.Next() {
		chunk := stream.Current()
		answer.AddChunk(chunk)
		if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content == "" {
			continue
		}
		if !yield(proxias.LLMChunk{Text: chunk.Choices[0].Delta.Content}, nil) {
			return
		}
	}
	if err := stream.Err(); err != nil {
		yield(proxias.LLMChunk{}, err)
		return
	}
	if len(answer.Choices) == 0 {
		return
	}
	choice := answer.Choices[0]
	if !m.yieldToolCalls(choice.Message, yield) {
		return
	}
	stop := m.mapStopReason(choice.FinishReason)
	if stop != "" {
		yield(proxias.LLMChunk{StopReason: stop}, nil)
	}
}

// yieldToolCalls hands out the function calls of the message, and returns
// false when the reader stopped, or the arguments of one are not JSON (its
// error handed out).
func (m LLM) yieldToolCalls(
	message openai.ChatCompletionMessage,
	yield func(proxias.LLMChunk, error) bool,
) bool {
	for _, call := range message.ToolCalls {
		if call.Type != "function" {
			continue
		}
		var input map[string]any
		if call.Function.Arguments != "" {
			arguments := []byte(call.Function.Arguments)
			if err := json.Unmarshal(arguments, &input); err != nil {
				yield(
					proxias.LLMChunk{},
					fmt.Errorf(
						"openai: the arguments of %s are not "+
							"a JSON object: %w",
						call.Function.Name,
						err,
					),
				)
				return false
			}
		}
		toolCall := proxias.LLMChunk{
			ToolCall: &proxias.LLMToolCall{
				ID:    call.ID,
				Name:  call.Function.Name,
				Input: input,
			},
		}
		if !yield(toolCall, nil) {
			return false
		}
	}
	return true
}

// mapStopReason maps why the model stopped; it did not when it asks for tool
// calls (or a function, the legacy call).
func (m LLM) mapStopReason(finishReason string) proxias.StopReason {
	switch finishReason {
	case "tool_calls", "function_call":
		return ""
	case "length":
		return proxias.StopReasonLength
	case "content_filter":
		return proxias.StopReasonContentFilter
	}
	return proxias.StopReasonEndTurn
}

// messages is the conversation as the API's, in order: system messages as
// system messages, the user's text as user messages, the results of tools as
// tool messages, and the assistant's text and tool calls as its messages.
// Thinking
// parts are left out, and empty texts.
func (m LLM) messages(conversation []proxias.LLMMessage) []openai.ChatCompletionMessageParamUnion {
	// Consecutive messages of the assistant are one: the API wants its tool
	// calls in one message, before their results.
	var merged []proxias.LLMMessage
	for _, message := range conversation {
		if message.Role != proxias.LLMMessageRoleAssistant {
			merged = append(merged, message)
			continue
		}
		last := len(merged) - 1
		if last >= 0 && merged[last].Role == proxias.LLMMessageRoleAssistant {
			merged[last].Parts = append(
				slices.Clip(merged[last].Parts),
				message.Parts...)
			continue
		}
		merged = append(merged, message)
	}
	params := make([]openai.ChatCompletionMessageParamUnion, 0, len(merged))
	for _, message := range merged {
		if message.Role == proxias.LLMMessageRoleSystem {
			var text []string
			for _, part := range message.Parts {
				if part.Text != "" {
					text = append(text, part.Text)
				}
			}
			if len(text) > 0 {
				params = append(params, openai.SystemMessage(strings.Join(text, "\n\n")))
			}
			continue
		}
		if message.Role == proxias.LLMMessageRoleAssistant {
			var (
				text      []string
				reasoning []string
				toolCalls []openai.ChatCompletionMessageToolCallUnionParam
			)
			for _, part := range message.Parts {
				switch {
				case part.Signature != "" || part.RedactedThinking != "":
					// Another vendor's thinking (Claude's): meaningless here.
				case part.Thinking != "":
					reasoning = append(reasoning, part.Thinking)
				case part.ToolCall != nil:
					input := part.ToolCall.Input
					if input == nil {
						input = map[string]any{}
					}
					// The input of a call was JSON.
					arguments, _ := json.Marshal(input)
					function := openai.
						ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      part.ToolCall.Name,
						Arguments: string(arguments),
					}
					call := openai.ChatCompletionMessageFunctionToolCallParam{
						ID:       part.ToolCall.ID,
						Function: function,
					}
					toolCalls = append(
						toolCalls,
						openai.ChatCompletionMessageToolCallUnionParam{
							OfFunction: &call,
						},
					)
				case part.Text != "":
					text = append(text, part.Text)
				}
			}
			if len(text) == 0 && len(toolCalls) == 0 {
				continue
			}
			assistant := openai.ChatCompletionAssistantMessageParam{
				ToolCalls: toolCalls,
			}
			if len(text) > 0 {
				content := openai.String(strings.Join(text, "\n\n"))
				assistant.Content = openai.
					ChatCompletionAssistantMessageParamContentUnion{
					OfString: content,
				}
			}
			// The model's reasoning is given back as the server sent it.
			if len(reasoning) > 0 {
				assistant.SetExtraFields(map[string]any{
					"reasoning_content": strings.Join(reasoning, "\n\n"),
				})
			}
			params = append(
				params,
				openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant},
			)
			continue
		}
		var text []string
		for _, part := range message.Parts {
			switch {
			case part.ToolResult != nil:
				if len(text) > 0 {
					params = append(params, openai.UserMessage(strings.Join(text, "\n\n")))
					text = nil
				}
				params = append(params, openai.ToolMessage(part.ToolResult.Content, part.ToolResult.CallID))
			case part.Text != "":
				text = append(text, part.Text)
			}
		}
		if len(text) > 0 {
			params = append(params, openai.UserMessage(strings.Join(text, "\n\n")))
		}
	}
	return params
}

// parameters is a JSON Schema of an object as a function's parameters: an
// action without one takes none.
func (m LLM) parameters(schema map[string]any) shared.FunctionParameters {
	if schema == nil {
		return shared.FunctionParameters{
			"type":       "object",
			"properties": map[string]any{},
		}
	}
	return shared.FunctionParameters(schema)
}
