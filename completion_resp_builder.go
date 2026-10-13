package proxias

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

// ChatCompletionBuilder makes the ChatCompletions that answer a request without a model: a text, or calls of tools
// for the harness to run. Give one to NewChatCompletionResponse, which streams it if the request asked for a
// stream. The completions of a builder share its model, ID and creation time.
type ChatCompletionBuilder struct {
	id      string
	model   string
	created int64
}

// NewChatCompletionBuilder returns a builder of the completions that answer req.
func NewChatCompletionBuilder(req ChatCompletionRequest) ChatCompletionBuilder {
	var id [12]byte
	_, _ = rand.Read(id[:]) // crypto/rand.Read does not fail since Go 1.24
	return ChatCompletionBuilder{
		id:      "chatcmpl-" + hex.EncodeToString(id[:]),
		model:   req.Model,
		created: time.Now().Unix(),
	}
}

// BuildText is the completion that answers with text.
func (b ChatCompletionBuilder) BuildText(text string) ChatCompletion {
	return b.completion(ChatMessage{
		Role:    ChatMessageRoleAssistant,
		Content: ChatContent{{Type: ChatContentPartTypeText, Text: text}},
	}, ChatFinishReasonStop)
}

// BuildToolCalls is the completion that asks the harness to run calls of its tools.
func (b ChatCompletionBuilder) BuildToolCalls(calls ...ChatToolCall) ChatCompletion {
	return b.completion(ChatMessage{Role: ChatMessageRoleAssistant, ToolCalls: calls}, ChatFinishReasonToolCalls)
}

func (b ChatCompletionBuilder) completion(message ChatMessage, finish ChatFinishReason) ChatCompletion {
	return ChatCompletion{
		ID: b.id, Object: "chat.completion", Created: b.created, Model: b.model,
		Choices: []ChatCompletionChoice{{Message: message, FinishReason: finish}},
	}
}
