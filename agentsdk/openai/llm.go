// Package openai is an OpenAI model for agentsdk, through the official SDK and its Chat Completions API: an
// agentsdk.LLM (NewLLM). The host builds the client (its key, timeout, retries), and chooses the model.
package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"

	"github.com/morebec/proxias/agentsdk"
)

// DefaultMaxTokens is the tokens an answer may take.
const DefaultMaxTokens = 4096

// LLM is an OpenAI model as an agentsdk.LLM. Make it with NewLLM.
type LLM struct {
	client    openai.Client
	model     string
	maxTokens int64
}

var _ agentsdk.LLM = LLM{}

// NewLLM returns the model of the client as an agentsdk.LLM: the system prompt is system messages, the conversation is
// the API's (tool calls in the assistant's messages, their results as tool messages), and the answer is read whole,
// then handed out as chunks. A model's thinking (another vendor's) is not given to it: the API takes none.
func NewLLM(client openai.Client, model string) LLM {
	return LLM{client: client, model: model, maxTokens: DefaultMaxTokens}
}

// Complete asks the model, and hands out its answer: its text, and its tool calls. The SDK's errors are returned as
// they are (errors.As an *openai.Error).
func (m LLM) Complete(ctx context.Context, req agentsdk.LLMRequest) iter.Seq2[agentsdk.LLMChunk, error] {
	return func(yield func(agentsdk.LLMChunk, error) bool) {
		params := openai.ChatCompletionNewParams{
			Model:               m.model,
			MaxCompletionTokens: openai.Int(m.maxTokens),
			Messages:            m.messages(req.System, req.Messages),
		}
		for _, tool := range req.Tools {
			params.Tools = append(params.Tools, openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
				Name:        tool.ID,
				Description: openai.String(tool.Description),
				Parameters:  m.parameters(tool.InputSchema),
			}))
		}

		completion, err := m.client.Chat.Completions.New(ctx, params)
		if err != nil {
			yield(agentsdk.LLMChunk{}, err)
			return
		}
		if len(completion.Choices) == 0 {
			return
		}
		message := completion.Choices[0].Message
		if message.Content != "" && !yield(agentsdk.LLMChunk{Text: message.Content}, nil) {
			return
		}
		for _, call := range message.ToolCalls {
			if call.Type != "function" {
				continue
			}
			var input map[string]any
			if call.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(call.Function.Arguments), &input); err != nil {
					yield(agentsdk.LLMChunk{}, fmt.Errorf("openai: the arguments of %s are not a JSON object: %w", call.Function.Name, err))
					return
				}
			}
			if !yield(agentsdk.LLMChunk{ToolCall: &agentsdk.LLMToolCall{ID: call.ID, Name: call.Function.Name, Input: input}}, nil) {
				return
			}
		}
	}
}

// messages is the system prompt and the conversation as the API's: a system message per block of the system prompt,
// the user's text as user messages and the results of tools as tool messages, in order, and the assistant's text and
// tool calls as its messages. Thinking parts are left out.
func (m LLM) messages(system []string, conversation []agentsdk.LLMMessage) []openai.ChatCompletionMessageParamUnion {
	params := make([]openai.ChatCompletionMessageParamUnion, 0, len(system)+len(conversation))
	for _, text := range system {
		params = append(params, openai.SystemMessage(text))
	}
	for _, message := range conversation {
		if message.Role == agentsdk.RoleAssistant {
			var (
				text      []string
				toolCalls []openai.ChatCompletionMessageToolCallUnionParam
			)
			for _, part := range message.Parts {
				switch {
				case part.ToolCall != nil:
					input := part.ToolCall.Input
					if input == nil {
						input = map[string]any{}
					}
					arguments, _ := json.Marshal(input) // the input of a call was JSON
					toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
						ID:       part.ToolCall.ID,
						Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{Name: part.ToolCall.Name, Arguments: string(arguments)},
					}})
				case part.Text != "":
					text = append(text, part.Text)
				}
			}
			if len(text) == 0 && len(toolCalls) == 0 {
				continue
			}
			assistant := openai.ChatCompletionAssistantMessageParam{ToolCalls: toolCalls}
			if len(text) > 0 {
				assistant.Content = openai.ChatCompletionAssistantMessageParamContentUnion{OfString: openai.String(strings.Join(text, "\n\n"))}
			}
			params = append(params, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant})
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

// parameters is a JSON Schema of an object as a function's parameters: an action without one takes none.
func (m LLM) parameters(schema map[string]any) shared.FunctionParameters {
	if schema == nil {
		return shared.FunctionParameters{"type": "object", "properties": map[string]any{}}
	}
	return shared.FunctionParameters(schema)
}
