package proxias

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// This file is the request side of the model of Proxias: the OpenAI chat completions API, as it goes over the
// wire. A harness speaks it, so a proxy that holds it as is loses nothing of a request that it forwards
// (images, tool calls, cache breakpoints...), and an adapter for another provider maps to and from it at its
// edge. Fields that a type does not name are kept in its Extra, and written back. The response side is in
// completion_resp.go.

// ChatCompletionRequest is a chat completion request as the harness sent it: the body of OpenAI's chat
// completions API, of which it names the fields that Proxias and its adapters read. The others are in Extra.
type ChatCompletionRequest struct {
	// Model is the model that the harness asks for.
	Model string `json:"model,omitempty"`

	// Stream tells that the harness wants the answer streamed.
	Stream bool `json:"stream,omitempty"`

	// StreamOptions are the options of a streamed answer, if the harness set any: a request without stream
	// options must not gain some, as OpenAI refuses them without a stream.
	StreamOptions *ChatStreamOptions `json:"stream_options,omitempty"`

	// Messages are the messages said so far, the one to complete last.
	Messages []ChatMessage `json:"messages"`

	// Tools are the tools that the harness offers to the model.
	Tools []ChatTool `json:"tools,omitempty"`

	// ToolChoice is the tool_choice of the request as it is: a string, or an object.
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`

	// ParallelToolCalls is nil if the harness did not say.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	// MaxTokens and MaxCompletionTokens limit the answer. The second is the current name of the first.
	MaxTokens           int64 `json:"max_tokens,omitempty"`
	MaxCompletionTokens int64 `json:"max_completion_tokens,omitempty"`

	// Stop is the stop of the request as it is: a string, or a list of strings.
	Stop json.RawMessage `json:"stop,omitempty"`

	// Header is the headers of the HTTP request that carried it, if it came over HTTP. They hold what the
	// body does not: the credentials, the tenant, a trace to follow. They are secrets, never to log.
	Header http.Header `json:"-"`

	// Extra are the fields of the JSON object that the type does not name: they are kept, and written back.
	Extra map[string]json.RawMessage `json:"-"`
}

// MaxOutputTokens is the limit of the answer, or 0 if the harness set none.
func (r ChatCompletionRequest) MaxOutputTokens() int64 {
	if r.MaxCompletionTokens > 0 {
		return r.MaxCompletionTokens
	}

	return r.MaxTokens
}

// UsageRequested tells whether the harness asked for the usage in a streamed answer.
func (r ChatCompletionRequest) UsageRequested() bool {
	return r.StreamOptions != nil && r.StreamOptions.IncludeUsage
}

// LastMessage is the last of the messages, the one to complete, or the zero ChatMessage if there are none.
func (r ChatCompletionRequest) LastMessage() ChatMessage {
	if len(r.Messages) == 0 {
		return ChatMessage{}
	}
	return r.Messages[len(r.Messages)-1]
}

// ChatStreamOptions are the options of a streamed answer.
type ChatStreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`

	// Extra are the fields of the JSON object that the type does not name: they are kept, and written back.
	Extra map[string]json.RawMessage `json:"-"`
}

// ChatMessage is one message of a ChatCompletionRequest, or the message of an answer.
type ChatMessage struct {
	Role ChatMessageRole `json:"role"`
	// Content is null for an assistant message that only calls tools, which is what the response must say too.
	Content ChatContent `json:"content"`
	// ToolCalls are the calls that an assistant message asks for.
	ToolCalls []ChatToolCall `json:"tool_calls,omitempty"`
	// ToolCallID is the call that a tool message answers.
	ToolCallID string `json:"tool_call_id,omitempty"`
	// Name is the name of the user or model that sent the message, if applicable.
	Name string `json:"name,omitempty"`

	// Extra are the fields of the JSON object that the type does not name: they are kept, and written back.
	Extra map[string]json.RawMessage `json:"-"`
}

// HasToolRole tells whether the message holds the result of a tool (or, formerly, of a function) that ran,
// as opposed to a message of a user or of a model.
func (m ChatMessage) HasToolRole() bool {
	return m.Role == ChatMessageRoleTool || m.Role == ChatMessageRoleFunction
}

// ChatMessageRole defines the role of a message in the chat completion.
// In other words it allows indicating who generated a particular message in the chat history.
type ChatMessageRole string

const (
	// ChatMessageRoleUser represents a user message in the chat completion.
	// Purpose: Contains the prompt, questions, or instructions sent by a human user.
	// Origin: Provided by the end-user interacting with an application/harness.
	ChatMessageRoleUser ChatMessageRole = "user"

	// ChatMessageRoleSystem represents a system message in the chat completion.
	// See ChatMessageRoleDeveloper for more details.
	ChatMessageRoleSystem ChatMessageRole = "system"

	// ChatMessageRoleDeveloper formally "system", represents an internal message.
	// Purpose: indicates that a message sets the persona, behavior, boundaries, and global context of the model.
	// Essentially guides the model's tone or dicatates formatting rules (e.g. "always return JSON"),
	// dictates guardrails, constraints, or safety measures for the model's behavior.
	// In most models it carries the highest level of authority over the model's behavior.
	// Origin: Usually set by harness/application developers, hidden from users.
	ChatMessageRoleDeveloper ChatMessageRole = "developer"

	// ChatMessageRoleAssistant represents an assistant message in the chat completion.
	// Purpose: Store the model's generated responses. Used to feed conversation history back to the model.
	// Because chat completion APIs are generally stateless, this provides a mechanism to maintain conversation
	// context across multiple turns.
	// Can also be used for few-shots prompting to mock how user and assistant turns to show the model how it should behave.
	// Origin: Generated by the model in response to user and system messages, or by backend code/harnesses when few-shots prompting is used.
	ChatMessageRoleAssistant ChatMessageRole = "assistant"

	// ChatMessageRoleFunction represents a function message in the chat completion.
	// See ChatMessageRoleTool for more details.
	ChatMessageRoleFunction ChatMessageRole = "function"

	// ChatMessageRoleTool represents a tool message in the chat completion.
	// Purpose: Passes the output/results of an external function back to the model after a tool call.
	// When a model calls an external tool, it stops generating text responses and outputs
	// "tool calls" requests instead. Code runs these tools locally and passes the output back to the model.
	// using the tool role, so the model can read it and formulate its final answer.
	// Origin: Executed by backend code.
	ChatMessageRoleTool ChatMessageRole = "tool"
)

// ChatContent is the content of a message: a list of parts. On the wire it is a string, which is one
// text part, or a list of parts, or null, which is no part.
type ChatContent []ChatContentPart

// Text is the text parts of the content, joined by new lines.
func (c ChatContent) Text() string {
	var texts []string
	for _, part := range c {
		if part.Type == ChatContentPartTypeText && part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// UnmarshalJSON implements json.Unmarshaler.
func (c *ChatContent) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	switch {
	case string(data) == "null":
		*c = nil
		return nil
	case len(data) > 0 && data[0] == '"':
		var text string
		if err := json.Unmarshal(data, &text); err != nil {
			return err
		}
		*c = ChatContent{{Type: ChatContentPartTypeText, Text: text}}
		return nil
	case len(data) > 0 && data[0] == '[':
		return json.Unmarshal(data, (*[]ChatContentPart)(c))
	}
	return fmt.Errorf("unsupported content: expected a string, a list of parts or null")
}

// MarshalJSON implements json.Marshaler. A lone text part is a string, as the harnesses that sent one expect,
// unless it has a breakpoint or fields that a string cannot hold.
func (c ChatContent) MarshalJSON() ([]byte, error) {
	switch {
	case len(c) == 0:
		return []byte("null"), nil
	case len(c) == 1 && c[0].Type == ChatContentPartTypeText && c[0].CacheControl == nil && c[0].Extra == nil:
		return json.Marshal(c[0].Text)
	}
	return json.Marshal([]ChatContentPart(c))
}

// ChatContentPartType is the type of a ChatContentPart.
type ChatContentPartType string

const (
	// ChatContentPartTypeText is a part of text.
	ChatContentPartTypeText ChatContentPartType = "text"
	// ChatContentPartTypeImageURL is an image, by address or as a data URL.
	ChatContentPartTypeImageURL ChatContentPartType = "image_url"
)

// ChatContentPart is a part of the content of a message.
type ChatContentPart struct {
	Type ChatContentPartType `json:"type"`
	Text string              `json:"text,omitempty"`
	// ImageURL is the image of a ChatContentPartTypeImageURL part.
	ImageURL *ChatImageURL `json:"image_url,omitempty"`
	// CacheControl is a cache breakpoint. It is not part of OpenAI's API, but harnesses that talk to Claude
	// through an OpenAI-compatible endpoint send it, and it must reach the provider that understands it.
	CacheControl *ChatCacheControl `json:"cache_control,omitempty"`

	// Extra are the fields of the JSON object that the type does not name: they are kept, and written back.
	Extra map[string]json.RawMessage `json:"-"`
}

// ChatImageURL is the image of a part: a web address, or a data URL (data:image/png;base64,...).
type ChatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`

	// Extra are the fields of the JSON object that the type does not name: they are kept, and written back.
	Extra map[string]json.RawMessage `json:"-"`
}

// ChatCacheControl marks the end of a part that the provider may cache.
type ChatCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// ChatToolCall is a call of a function that a model asks for, in a message.
type ChatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ChatFunctionCall `json:"function"`

	// Extra are the fields of the JSON object that the type does not name: they are kept, and written back.
	Extra map[string]json.RawMessage `json:"-"`
}

// ChatFunctionCall is the function of a ChatToolCall: its name, and its arguments as a JSON text.
type ChatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`

	// Extra are the fields of the JSON object that the type does not name: they are kept, and written back.
	Extra map[string]json.RawMessage `json:"-"`
}

// ChatTool is a tool that the harness offers to the model.
type ChatTool struct {
	Type     string       `json:"type"`
	Function ChatFunction `json:"function"`

	// Extra are the fields of the JSON object that the type does not name: they are kept, and written back.
	Extra map[string]json.RawMessage `json:"-"`
}

// ChatFunction is the function of a ChatTool, its Parameters being a JSON Schema.
type ChatFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`

	// Extra are the fields of the JSON object that the type does not name: they are kept, and written back.
	Extra map[string]json.RawMessage `json:"-"`
}

// UnmarshalJSON implements json.Unmarshaler. The fields the type does not name are kept in Extra.
func (r *ChatCompletionRequest) UnmarshalJSON(data []byte) error {
	type plain ChatCompletionRequest
	var p plain
	extra, err := unmarshalPlain(data, &p)
	if err != nil {
		return err
	}
	*r = ChatCompletionRequest(p)
	r.Extra = extra
	return nil
}

// MarshalJSON implements json.Marshaler. The fields of Extra are written, unless a typed field has the same name.
func (r ChatCompletionRequest) MarshalJSON() ([]byte, error) {
	type plain ChatCompletionRequest
	return marshalPlain(plain(r), r.Extra)
}

// UnmarshalJSON implements json.Unmarshaler. The fields the type does not name are kept in Extra.
func (r *ChatStreamOptions) UnmarshalJSON(data []byte) error {
	type plain ChatStreamOptions
	var p plain
	extra, err := unmarshalPlain(data, &p)
	if err != nil {
		return err
	}
	*r = ChatStreamOptions(p)
	r.Extra = extra
	return nil
}

// MarshalJSON implements json.Marshaler. The fields of Extra are written, unless a typed field has the same name.
func (r ChatStreamOptions) MarshalJSON() ([]byte, error) {
	type plain ChatStreamOptions
	return marshalPlain(plain(r), r.Extra)
}

// UnmarshalJSON implements json.Unmarshaler. The fields the type does not name are kept in Extra.
func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	type plain ChatMessage
	var p plain
	extra, err := unmarshalPlain(data, &p)
	if err != nil {
		return err
	}
	*m = ChatMessage(p)
	m.Extra = extra
	return nil
}

// MarshalJSON implements json.Marshaler. The fields of Extra are written, unless a typed field has the same name.
func (m ChatMessage) MarshalJSON() ([]byte, error) {
	type plain ChatMessage
	return marshalPlain(plain(m), m.Extra)
}

// UnmarshalJSON implements json.Unmarshaler. The fields the type does not name are kept in Extra.
func (r *ChatContentPart) UnmarshalJSON(data []byte) error {
	type plain ChatContentPart
	var p plain
	extra, err := unmarshalPlain(data, &p)
	if err != nil {
		return err
	}
	*r = ChatContentPart(p)
	r.Extra = extra
	return nil
}

// MarshalJSON implements json.Marshaler. The fields of Extra are written, unless a typed field has the same name.
func (r ChatContentPart) MarshalJSON() ([]byte, error) {
	type plain ChatContentPart
	return marshalPlain(plain(r), r.Extra)
}

// UnmarshalJSON implements json.Unmarshaler. The fields the type does not name are kept in Extra.
func (r *ChatImageURL) UnmarshalJSON(data []byte) error {
	type plain ChatImageURL
	var p plain
	extra, err := unmarshalPlain(data, &p)
	if err != nil {
		return err
	}
	*r = ChatImageURL(p)
	r.Extra = extra
	return nil
}

// MarshalJSON implements json.Marshaler. The fields of Extra are written, unless a typed field has the same name.
func (r ChatImageURL) MarshalJSON() ([]byte, error) {
	type plain ChatImageURL
	return marshalPlain(plain(r), r.Extra)
}

// UnmarshalJSON implements json.Unmarshaler. The fields the type does not name are kept in Extra.
func (r *ChatToolCall) UnmarshalJSON(data []byte) error {
	type plain ChatToolCall
	var p plain
	extra, err := unmarshalPlain(data, &p)
	if err != nil {
		return err
	}
	*r = ChatToolCall(p)
	r.Extra = extra
	return nil
}

// MarshalJSON implements json.Marshaler. The fields of Extra are written, unless a typed field has the same name.
func (r ChatToolCall) MarshalJSON() ([]byte, error) {
	type plain ChatToolCall
	return marshalPlain(plain(r), r.Extra)
}

// UnmarshalJSON implements json.Unmarshaler. The fields the type does not name are kept in Extra.
func (r *ChatFunctionCall) UnmarshalJSON(data []byte) error {
	type plain ChatFunctionCall
	var p plain
	extra, err := unmarshalPlain(data, &p)
	if err != nil {
		return err
	}
	*r = ChatFunctionCall(p)
	r.Extra = extra
	return nil
}

// MarshalJSON implements json.Marshaler. The fields of Extra are written, unless a typed field has the same name.
func (r ChatFunctionCall) MarshalJSON() ([]byte, error) {
	type plain ChatFunctionCall
	return marshalPlain(plain(r), r.Extra)
}

// UnmarshalJSON implements json.Unmarshaler. The fields the type does not name are kept in Extra.
func (r *ChatTool) UnmarshalJSON(data []byte) error {
	type plain ChatTool
	var p plain
	extra, err := unmarshalPlain(data, &p)
	if err != nil {
		return err
	}
	*r = ChatTool(p)
	r.Extra = extra
	return nil
}

// MarshalJSON implements json.Marshaler. The fields of Extra are written, unless a typed field has the same name.
func (r ChatTool) MarshalJSON() ([]byte, error) {
	type plain ChatTool
	return marshalPlain(plain(r), r.Extra)
}

// UnmarshalJSON implements json.Unmarshaler. The fields the type does not name are kept in Extra.
func (r *ChatFunction) UnmarshalJSON(data []byte) error {
	type plain ChatFunction
	var p plain
	extra, err := unmarshalPlain(data, &p)
	if err != nil {
		return err
	}
	*r = ChatFunction(p)
	r.Extra = extra
	return nil
}

// MarshalJSON implements json.Marshaler. The fields of Extra are written, unless a typed field has the same name.
func (r ChatFunction) MarshalJSON() ([]byte, error) {
	type plain ChatFunction
	return marshalPlain(plain(r), r.Extra)
}
