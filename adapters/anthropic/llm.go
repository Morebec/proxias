// Package anthropic is Claude for proxias, through the official SDK: an
// proxias.LLM (NewLLM), and a counter of what the calls cost (UsageCounter).
// The host builds the client (its key, timeout, retries), and chooses the
// model.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/morebec/proxias"
)

// DefaultMaxTokens is the tokens an answer may take.
const DefaultMaxTokens = 4096

// LLM is Claude as an proxias.LLM. Make it with NewLLM.
type LLM struct {
	client anthropic.Client
	model  anthropic.Model
	// MaxTokens is the tokens an answer may take: DefaultMaxTokens by default.
	MaxTokens int64
	// InstructionsTTL is how long the tools and the system prompt stay
	// cached: Claude's default (5 minutes) when empty, or
	// anthropic.CacheControlEphemeralTTLTTL1h for instructions that every
	// conversation shares and that a person may come back to after a pause.
	// Unused when Uncached.
	InstructionsTTL anthropic.CacheControlEphemeralTTL
	// Thinking configures Claude's thinking, sent as is when set: for example
	// adaptive thinking that returns a summary of it, {OfAdaptive:
	// &anthropic.ThinkingConfigAdaptiveParam{Display:
	// anthropic.ThinkingConfigAdaptiveDisplaySummarized}}. Unset, the model's
	// default applies. Its thinking blocks are handed out, and given back
	// signed as they came.
	Thinking anthropic.ThinkingConfigParamUnion
	// Effort is how much Claude spends on an answer, its thinking included:
	// the model's default when empty. A request's own Effort overrides it.
	// Models before Claude 4.6 refuse it.
	Effort proxias.Effort
	// Uncached sends nothing to cache, for calls whose start no later call
	// shares (a one-off question): a cache write costs more than reading the
	// input once.
	Uncached bool
	// Buffered has Complete read the answer whole, then hand it out, instead
	// of streaming it: by default its text and its thinking are handed out as
	// Claude writes them, each of its other blocks (thinking, tool calls) once
	// it is whole.
	Buffered bool
}

var _ proxias.LLM = LLM{}

// NewLLM returns the model of the client as an proxias.LLM: unless Uncached,
// the tools are cached, then the system prompt, each at its own breakpoint,
// and so is the conversation so far (its automatic breakpoint moves to its end
// at every call, so the turns of a loop read it from the cache); the conversation is Claude's (tool
// calls as tool_use blocks, their results as tool_result blocks), and the
// answer is streamed (Buffered reads it whole).
func NewLLM(client anthropic.Client, model anthropic.Model) LLM {
	return LLM{
		client:    client,
		model:     model,
		MaxTokens: DefaultMaxTokens,
	}
}

// Complete asks Claude, and hands out its answer: its thinking, its text, and
// its tool calls. The
// SDK's errors are returned as they are (errors.As an *anthropic.Error).
func (m LLM) Complete(ctx context.Context, req proxias.LLMRequest) iter.Seq2[proxias.LLMChunk, error] {
	return func(yield func(proxias.LLMChunk, error) bool) {
		// The system messages that open the conversation are Claude's system
		// prompt; the API has no system turns.
		var system []anthropic.TextBlockParam
		instructions := 0
		for _, message := range req.Messages {
			if message.Role != proxias.LLMMessageRoleSystem {
				break
			}
			for _, part := range message.Parts {
				if part.Text != "" {
					system = append(
						system,
						anthropic.TextBlockParam{Text: part.Text},
					)
				}
			}
			instructions++
		}

		params := anthropic.MessageNewParams{
			Model:     m.modelOf(req),
			MaxTokens: m.MaxTokens,
			System:    system,
			Messages:  m.mapMessages(req.Messages[instructions:], len(req.Tools) > 0),
			Thinking:  m.Thinking,
			OutputConfig: anthropic.OutputConfigParam{
				Effort: anthropic.OutputConfigEffort(m.effortOf(req)),
			},
		}
		for _, tool := range req.Tools {
			t := anthropic.ToolParam{
				Name:        tool.ID,
				Description: anthropic.String(tool.Description),
				InputSchema: m.inputSchema(tool.InputSchema),
			}
			params.Tools = append(
				params.Tools,
				anthropic.ToolUnionParam{OfTool: &t},
			)
		}
		if !m.Uncached {
			m.markCachePoints(&params)
		}

		if m.Buffered {
			m.complete(ctx, params, yield)
			return
		}
		m.stream(ctx, params, yield)
	}
}

// markCachePoints caches the tools, the system prompt and the conversation, each
// at its own breakpoint: the tools rarely change, the system prompt now and
// then, the conversation at every call, so a change to one does not write the
// ones before it again. The tools and the system prompt are kept for
// InstructionsTTL, the conversation for Claude's default.
func (m LLM) markCachePoints(params *anthropic.MessageNewParams) {
	instructions := anthropic.NewCacheControlEphemeralParam()
	instructions.TTL = m.InstructionsTTL
	if len(params.Tools) > 0 {
		params.Tools[len(params.Tools)-1].OfTool.CacheControl = instructions
	}
	if len(params.System) > 0 {
		params.System[len(params.System)-1].CacheControl = instructions
	}
	params.CacheControl = anthropic.NewCacheControlEphemeralParam()
}

// modelOf is the request's model, or the LLM's when it names none.
func (m LLM) modelOf(req proxias.LLMRequest) anthropic.Model {
	if req.Model != "" {
		return req.Model
	}
	return m.model
}

// effortOf is the request's effort, or the LLM's when it names none.
func (m LLM) effortOf(req proxias.LLMRequest) proxias.Effort {
	if req.Effort != "" {
		return req.Effort
	}
	return m.Effort
}

// complete reads Claude's answer whole, then hands out its blocks.
func (m LLM) complete(
	ctx context.Context,
	params anthropic.MessageNewParams,
	yield func(proxias.LLMChunk, error) bool,
) {
	message, err := m.client.Messages.New(ctx, params)
	if err != nil {
		yield(proxias.LLMChunk{}, err)
		return
	}
	for _, block := range message.Content {
		chunk, ok, err := m.chunkOf(block)
		if err != nil {
			yield(proxias.LLMChunk{}, err)
			return
		}
		if !ok {
			continue
		}
		if !yield(chunk, nil) {
			return
		}
	}
	stop := m.mapStopReason(message.StopReason)
	if stop != "" {
		yield(proxias.LLMChunk{StopReason: stop}, nil)
	}
}

// stream reads Claude's answer as it comes: each piece of its text and of its
// thinking as it is written, each other block once it is whole.
func (m LLM) stream(
	ctx context.Context,
	params anthropic.MessageNewParams,
	yield func(proxias.LLMChunk, error) bool,
) {
	stream := m.client.Messages.NewStreaming(ctx, params)
	defer stream.Close()

	var message anthropic.Message
	for stream.Next() {
		event := stream.Current()
		if err := message.Accumulate(event); err != nil {
			yield(proxias.LLMChunk{}, err)
			return
		}
		switch e := event.AsAny().(type) {
		case anthropic.ContentBlockDeltaEvent:
			chunk, ok := m.deltaChunkOf(e.Delta)
			if !ok {
				continue
			}
			if !yield(chunk, nil) {
				return
			}
		case anthropic.ContentBlockStopEvent:
			block := message.Content[e.Index]
			if block.Type == "text" {
				continue // its text is handed out already
			}
			chunk, ok, err := m.chunkOf(block)
			if err != nil {
				yield(proxias.LLMChunk{}, err)
				return
			}
			if !ok {
				continue
			}
			if !yield(chunk, nil) {
				return
			}
		}
	}
	if err := stream.Err(); err != nil {
		yield(proxias.LLMChunk{}, err)
		return
	}
	stop := m.mapStopReason(message.StopReason)
	if stop != "" {
		yield(proxias.LLMChunk{StopReason: stop}, nil)
	}
}

// deltaChunkOf is a piece of Claude's answer as a chunk: a piece of its text, or
// of its thinking (shown as it is written; the whole block, signed, follows);
// false for a piece that is none (empty, or of a tool call's input).
func (m LLM) deltaChunkOf(delta anthropic.RawContentBlockDeltaUnion) (proxias.LLMChunk, bool) {
	switch d := delta.AsAny().(type) {
	case anthropic.TextDelta:
		return proxias.LLMChunk{Text: d.Text}, d.Text != ""
	case anthropic.ThinkingDelta:
		return proxias.LLMChunk{ThinkingDelta: d.Thinking}, d.Thinking != ""
	}
	return proxias.LLMChunk{}, false
}

// chunkOf is a block of Claude's answer as a chunk, and false for a block
// that is none (an unknown type).
func (m LLM) chunkOf(block anthropic.ContentBlockUnion) (proxias.LLMChunk, bool, error) {
	var chunk proxias.LLMChunk
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
				return proxias.LLMChunk{}, false, fmt.Errorf(
					"anthropic: the input of %s is not a JSON object: %w",
					b.Name,
					err,
				)
			}
		}
		chunk.ToolCall = &proxias.LLMToolCall{
			ID:    b.ID,
			Name:  b.Name,
			Input: input,
		}
	default:
		return proxias.LLMChunk{}, false, nil
	}
	return chunk, true, nil
}

// mapStopReason maps why Claude stopped; it did not when it asks for tool
// calls.
func (m LLM) mapStopReason(stopReason anthropic.StopReason) proxias.StopReason {
	switch stopReason {
	case anthropic.StopReasonToolUse:
		return ""
	case anthropic.StopReasonMaxTokens:
		return proxias.StopReasonLength
	case anthropic.StopReasonRefusal:
		return proxias.StopReasonContentFilter
	}
	return proxias.StopReasonEndTurn
}

// mapMessages is the conversation as Claude's: consecutive messages of one
// role in one message, ending on a user turn as the API requires. The results
// of
// tools, and a system message after the first turn, are the user's (the latter
// tagged [system]). Without tools, the API refuses tool_use and tool_result
// blocks: tool calls and their results are left out. Thinking is given back
// only from the assistant turn whose tool calls the conversation ends by
// answering: Claude needs no other, and refuses a signature made in another
// call (under another system prompt, other tools, or other messages).
func (m LLM) mapMessages(conversation []proxias.LLMMessage, hasTools bool) []anthropic.MessageParam {
	var params []anthropic.MessageParam
	add := func(role anthropic.MessageParamRole, block anthropic.ContentBlockParamUnion) {
		last := len(params) - 1
		if last >= 0 && params[last].Role == role {
			params[last].Content = append(params[last].Content, block)
			return
		}
		params = append(params, anthropic.MessageParam{
			Role:    role,
			Content: []anthropic.ContentBlockParamUnion{block},
		})
	}
	// Without tools, no tool call is answered: no thinking is given back.
	first, end := 0, 0
	if hasTools {
		first, end = answeredTurn(conversation)
	}
	for i, message := range conversation {
		role := anthropic.MessageParamRoleUser
		if message.Role == proxias.LLMMessageRoleAssistant {
			role = anthropic.MessageParamRoleAssistant
		}
		for _, part := range message.Parts {
			switch {
			case part.Signature != "":
				if i < first || i >= end {
					// The API refuses thinking signed in another call.
					continue
				}
				add(role, anthropic.NewThinkingBlock(part.Signature, part.Thinking))
			case part.RedactedThinking != "":
				if i < first || i >= end {
					// The API refuses thinking signed in another call.
					continue
				}
				add(role, anthropic.NewRedactedThinkingBlock(part.RedactedThinking))
			case part.ToolCall != nil:
				if !hasTools {
					// The API refuses a tool_use block without tools.
					continue
				}
				input := part.ToolCall.Input
				if input == nil {
					input = map[string]any{}
				}
				add(role, anthropic.NewToolUseBlock(part.ToolCall.ID, input, part.ToolCall.Name))
			case part.ToolResult != nil:
				if !hasTools {
					// The API refuses a tool_result block without tools.
					continue
				}
				add(role, anthropic.NewToolResultBlock(part.ToolResult.CallID, part.ToolResult.Content, false))
			case part.Text != "" && message.Role == proxias.LLMMessageRoleSystem:
				add(role, anthropic.NewTextBlock("[system] "+part.Text))
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

// answeredTurn is the messages, first to end (excluded), of the assistant turn
// whose tool calls the conversation ends by answering: none when it does not
// end on tool results.
func answeredTurn(conversation []proxias.LLMMessage) (int, int) {
	end := len(conversation)
	for end > 0 && hasToolResult(conversation[end-1]) {
		end--
	}
	if end == len(conversation) {
		return 0, 0
	}

	first := end
	for first > 0 && conversation[first-1].Role == proxias.LLMMessageRoleAssistant {
		first--
	}
	return first, end
}

// hasToolResult tells whether the message gives the result of a tool call.
func hasToolResult(message proxias.LLMMessage) bool {
	for _, part := range message.Parts {
		if part.ToolResult != nil {
			return true
		}
	}
	return false
}

// inputSchema is a JSON Schema of an object as a tool's input schema: an action
// without one takes no parameter.
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
