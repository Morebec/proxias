package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/morebec/proxias"
)

// defaultMaxTokens is the limit of the answer when the harness sets none, which Claude requires.
const defaultMaxTokens = 4096

// mapChatCompletionRequestToMessageParams maps the chat completion request of the harness to a Messages API request.
//
// System and developer messages are the system prompt, a block for each text part. Tool results and what
// follows them are one user message, as the Messages API wants. Cache breakpoints (cache_control) of the
// parts are kept. Thinking is never sent back: chat completions have no place for it.
func mapChatCompletionRequestToMessageParams(req proxias.ChatCompletionRequest, model anthropicsdk.Model) (anthropicsdk.MessageNewParams, error) {
	params := anthropicsdk.MessageNewParams{
		Model:         model,
		MaxTokens:     defaultMaxTokens,
		StopSequences: mapStopSequences(req.Stop),
	}
	if limit := req.MaxOutputTokens(); limit > 0 {
		params.MaxTokens = limit
	}

	for i, m := range req.Messages {
		if m.Role == proxias.ChatMessageRoleSystem || m.Role == proxias.ChatMessageRoleDeveloper {
			for _, part := range m.Content {
				if part.Type == proxias.ChatContentPartTypeText && part.Text != "" {
					block := anthropicsdk.TextBlockParam{Text: part.Text}
					if part.CacheControl != nil {
						block.CacheControl = mapCacheControl(*part.CacheControl)
					}
					params.System = append(params.System, block)
				}
			}
			continue
		}

		blocks, err := mapContentBlocks(m)
		if err != nil {
			return anthropicsdk.MessageNewParams{}, fmt.Errorf("message %d: %w", i, err)
		}
		if len(blocks) == 0 {
			continue // an empty message, such as an assistant one with neither text nor tool call
		}

		role := anthropicsdk.MessageParamRoleUser
		if m.Role == proxias.ChatMessageRoleAssistant {
			role = anthropicsdk.MessageParamRoleAssistant
		}
		if last := len(params.Messages) - 1; last >= 0 && params.Messages[last].Role == role {
			params.Messages[last].Content = append(params.Messages[last].Content, blocks...)
		} else {
			params.Messages = append(params.Messages, anthropicsdk.MessageParam{Role: role, Content: blocks})
		}
	}

	for _, tool := range req.Tools {
		params.Tools = append(params.Tools, anthropicsdk.ToolUnionParam{OfTool: &anthropicsdk.ToolParam{
			Name:        tool.Function.Name,
			Description: anthropicsdk.String(tool.Function.Description),
			InputSchema: mapInputSchema(tool.Function.Parameters),
		}})
	}
	if len(params.Tools) > 0 {
		params.ToolChoice = mapToolChoice(req.ToolChoice, req.ParallelToolCalls)
	}
	return params, nil
}

// mapContentBlocks maps one message that is not a system one.
func mapContentBlocks(m proxias.ChatMessage) ([]anthropicsdk.ContentBlockParamUnion, error) {
	switch m.Role {
	case proxias.ChatMessageRoleTool:
		result := anthropicsdk.ToolResultBlockParam{ToolUseID: m.ToolCallID}
		if text := m.Content.Text(); text != "" {
			result.Content = []anthropicsdk.ToolResultBlockParamContentUnion{{OfText: &anthropicsdk.TextBlockParam{Text: text}}}
		}
		return []anthropicsdk.ContentBlockParamUnion{{OfToolResult: &result}}, nil

	case proxias.ChatMessageRoleAssistant:
		blocks := mapPartBlocks(m.Content)
		for _, call := range m.ToolCalls {
			input := json.RawMessage(call.Function.Arguments)
			if !json.Valid(input) || !strings.HasPrefix(strings.TrimSpace(call.Function.Arguments), "{") {
				input = json.RawMessage(`{}`) // the API takes an object
			}
			blocks = append(blocks, anthropicsdk.NewToolUseBlock(call.ID, input, call.Function.Name))
		}
		return blocks, nil

	case proxias.ChatMessageRoleUser:
		return mapPartBlocks(m.Content), nil
	}
	return nil, fmt.Errorf("unsupported role %q", m.Role)
}

// mapPartBlocks maps the parts of a content: text, and images by data URL or by address. Other parts are left
// out.
func mapPartBlocks(content proxias.ChatContent) []anthropicsdk.ContentBlockParamUnion {
	blocks := make([]anthropicsdk.ContentBlockParamUnion, 0, len(content))
	for _, part := range content {
		var block anthropicsdk.ContentBlockParamUnion
		switch {
		case part.Type == proxias.ChatContentPartTypeText && part.Text != "":
			block = anthropicsdk.NewTextBlock(part.Text)
		case part.Type == proxias.ChatContentPartTypeImageURL && part.ImageURL != nil:
			var ok bool
			if block, ok = mapImageBlock(part.ImageURL.URL); !ok {
				continue
			}
		default:
			continue
		}
		if cc := block.GetCacheControl(); cc != nil && part.CacheControl != nil {
			*cc = mapCacheControl(*part.CacheControl)
		}
		blocks = append(blocks, block)
	}
	return blocks
}

// mapImageBlock reads a data URL (data:image/png;base64,...) or a web address.
func mapImageBlock(url string) (anthropicsdk.ContentBlockParamUnion, bool) {
	if rest, ok := strings.CutPrefix(url, "data:"); ok {
		mediaType, data, found := strings.Cut(rest, ";base64,")
		if !found {
			return anthropicsdk.ContentBlockParamUnion{}, false
		}
		return anthropicsdk.NewImageBlockBase64(mediaType, data), true
	}
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return anthropicsdk.NewImageBlock(anthropicsdk.URLImageSourceParam{URL: url}), true
	}
	return anthropicsdk.ContentBlockParamUnion{}, false
}

// mapCacheControl maps a cache breakpoint. Claude knows the ephemeral one, for 5 minutes or an hour.
func mapCacheControl(cc proxias.ChatCacheControl) anthropicsdk.CacheControlEphemeralParam {
	if cc.TTL == "1h" {
		return anthropicsdk.CacheControlEphemeralParam{TTL: anthropicsdk.CacheControlEphemeralTTLTTL1h}
	}
	return anthropicsdk.NewCacheControlEphemeralParam()
}

// mapInputSchema keeps the JSON Schema of a tool as given: properties and required where the SDK wants them,
// everything else as extra fields.
func mapInputSchema(parameters map[string]any) anthropicsdk.ToolInputSchemaParam {
	schema := anthropicsdk.ToolInputSchemaParam{Properties: map[string]any{}}
	for key, value := range parameters {
		switch key {
		case "type":
		case "properties":
			schema.Properties = value
		case "required":
			if names, ok := value.([]any); ok {
				for _, name := range names {
					if name, ok := name.(string); ok {
						schema.Required = append(schema.Required, name)
					}
				}
			}
		default:
			if schema.ExtraFields == nil {
				schema.ExtraFields = map[string]any{}
			}
			schema.ExtraFields[key] = value
		}
	}
	return schema
}

// mapToolChoice keeps "none"; anything else is "auto": current models refuse a forced tool.
func mapToolChoice(raw json.RawMessage, parallelToolCalls *bool) anthropicsdk.ToolChoiceUnionParam {
	var choice string
	if json.Unmarshal(raw, &choice) == nil && choice == "none" {
		return anthropicsdk.ToolChoiceUnionParam{OfNone: &anthropicsdk.ToolChoiceNoneParam{}}
	}
	auto := anthropicsdk.ToolChoiceAutoParam{}
	if parallelToolCalls != nil && !*parallelToolCalls {
		auto.DisableParallelToolUse = anthropicsdk.Bool(true)
	}
	return anthropicsdk.ToolChoiceUnionParam{OfAuto: &auto}
}

// mapStopSequences reads a stop given as one string or as several. The API refuses blank ones.
func mapStopSequences(raw json.RawMessage) []string {
	var sequences []string
	var one string
	if json.Unmarshal(raw, &one) == nil {
		sequences = []string{one}
	} else {
		_ = json.Unmarshal(raw, &sequences)
	}

	var kept []string
	for _, sequence := range sequences {
		if strings.TrimSpace(sequence) != "" {
			kept = append(kept, sequence)
		}
	}
	return kept
}
