// Package anthropic is Claude for agentsdk, through the official SDK: an agentsdk.LLM (NewLLM), and a counter of what
// the calls cost (UsageCounter). The host builds the client (its key, timeout, retries), and chooses the model.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/morebec/proxias/agentsdk"
)

// DefaultMaxTokens is the tokens an answer may take.
const DefaultMaxTokens = 4096

// LLM is Claude as an agentsdk.LLM. Make it with NewLLM.
type LLM struct {
	client    anthropic.Client
	model     anthropic.Model
	maxTokens int64
	// Thinking configures Claude's thinking, sent as is when set: for example adaptive thinking that returns a summary
	// of it, {OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{Display: anthropic.ThinkingConfigAdaptiveDisplaySummarized}}.
	// Unset, the model's default applies. Its thinking blocks are handed out, and given back signed as they came.
	Thinking anthropic.ThinkingConfigParamUnion
}

var _ agentsdk.LLM = LLM{}

// NewLLM returns the model of the client as an agentsdk.LLM: the system prompt is cached (with the tools, which come
// before it), the conversation is Claude's (tool calls as tool_use blocks, their results as tool_result blocks), and the
// answer is read whole, then handed out as chunks.
func NewLLM(client anthropic.Client, model anthropic.Model) LLM {
	return LLM{client: client, model: model, maxTokens: DefaultMaxTokens}
}

// Complete asks Claude, and hands out its answer: its thinking, its text, and its tool calls. The
// SDK's errors are returned as they are (errors.As an *anthropic.Error).
func (m LLM) Complete(ctx context.Context, req agentsdk.LLMRequest) iter.Seq2[agentsdk.LLMChunk, error] {
	return func(yield func(agentsdk.LLMChunk, error) bool) {
		params := anthropic.MessageNewParams{Model: m.model, MaxTokens: m.maxTokens, Messages: m.messages(req.Messages), Thinking: m.Thinking}
		for _, text := range req.System {
			params.System = append(params.System, anthropic.TextBlockParam{Text: text})
		}
		if n := len(params.System); n > 0 {
			params.System[n-1].CacheControl = anthropic.NewCacheControlEphemeralParam()
		}
		for _, tool := range req.Tools {
			t := anthropic.ToolParam{Name: tool.ID, Description: anthropic.String(tool.Description), InputSchema: m.inputSchema(tool.InputSchema)}
			params.Tools = append(params.Tools, anthropic.ToolUnionParam{OfTool: &t})
		}

		message, err := m.client.Messages.New(ctx, params)
		if err != nil {
			yield(agentsdk.LLMChunk{}, err)
			return
		}
		for _, block := range message.Content {
			var chunk agentsdk.LLMChunk
			switch b := block.AsAny().(type) {
			case anthropic.ThinkingBlock:
				chunk.Thinking, chunk.Signature = b.Thinking, b.Signature
			case anthropic.RedactedThinkingBlock:
				chunk.RedactedThinking = b.Data
			case anthropic.TextBlock:
				chunk.Text = b.Text
			case anthropic.ToolUseBlock:
				var input map[string]any
				if len(b.Input) > 0 {
					if err := json.Unmarshal(b.Input, &input); err != nil {
						yield(agentsdk.LLMChunk{}, fmt.Errorf("anthropic: the input of %s is not a JSON object: %w", b.Name, err))
						return
					}
				}
				chunk.ToolCall = &agentsdk.LLMToolCall{ID: b.ID, Name: b.Name, Input: input}
			default:
				continue
			}
			if !yield(chunk, nil) {
				return
			}
		}
	}
}

// messages is the conversation as Claude's: consecutive messages of one role in one message, ending on a user turn as the API requires.
func (m LLM) messages(conversation []agentsdk.LLMMessage) []anthropic.MessageParam {
	var params []anthropic.MessageParam
	add := func(role anthropic.MessageParamRole, block anthropic.ContentBlockParamUnion) {
		if n := len(params); n > 0 && params[n-1].Role == role {
			params[n-1].Content = append(params[n-1].Content, block)
			return
		}
		params = append(params, anthropic.MessageParam{Role: role, Content: []anthropic.ContentBlockParamUnion{block}})
	}
	for _, message := range conversation {
		role := anthropic.MessageParamRoleUser
		if message.Role == agentsdk.RoleAssistant {
			role = anthropic.MessageParamRoleAssistant
		}
		for _, part := range message.Parts {
			switch {
			case part.Signature != "":
				add(role, anthropic.NewThinkingBlock(part.Signature, part.Thinking))
			case part.RedactedThinking != "":
				add(role, anthropic.NewRedactedThinkingBlock(part.RedactedThinking))
			case part.ToolCall != nil:
				input := part.ToolCall.Input
				if input == nil {
					input = map[string]any{}
				}
				add(role, anthropic.NewToolUseBlock(part.ToolCall.ID, input, part.ToolCall.Name))
			case part.ToolResult != nil:
				add(role, anthropic.NewToolResultBlock(part.ToolResult.CallID, part.ToolResult.Content, false))
			case part.Text != "":
				add(role, anthropic.NewTextBlock(part.Text))
			}
		}
	}
	if len(params) == 0 {
		add(anthropic.MessageParamRoleUser, anthropic.NewTextBlock("(nothing happened yet)"))
	}
	if params[len(params)-1].Role != anthropic.MessageParamRoleUser {
		add(anthropic.MessageParamRoleUser, anthropic.NewTextBlock("[no new event]"))
	}
	return params
}

// inputSchema is a JSON Schema of an object as a tool's input schema: an action without one takes no parameter.
func (m LLM) inputSchema(schema map[string]any) anthropic.ToolInputSchemaParam {
	param := anthropic.ToolInputSchemaParam{Properties: map[string]any{}}
	for key, value := range schema {
		switch key {
		case "type":
		case "properties":
			param.Properties = value
		case "required":
			switch required := value.(type) {
			case []string:
				param.Required = required
			case []any:
				for _, r := range required {
					param.Required = append(param.Required, fmt.Sprint(r))
				}
			}
		default:
			if param.ExtraFields == nil {
				param.ExtraFields = map[string]any{}
			}
			param.ExtraFields[key] = value
		}
	}
	return param
}
