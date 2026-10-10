package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/morebec/proxias"
)

// mapAnthropicMessageToCompletionResponse converts Claude's answer to a chat completion, as the OpenAI chat completions API
// answers.
func mapAnthropicMessageToCompletionResponse(message *anthropicsdk.Message) (proxias.CompletionResponse, error) {
	var text strings.Builder
	for _, block := range message.Content {
		if t, ok := block.AsAny().(anthropicsdk.TextBlock); ok {
			text.WriteString(t.Text)
		}
	}

	finishReason := "stop"
	switch message.StopReason {
	case anthropicsdk.StopReasonMaxTokens:
		finishReason = "length"
	case anthropicsdk.StopReasonRefusal:
		finishReason = "content_filter"
	}

	completion := messageCompletion{
		ID:      message.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   string(message.Model),
		Choices: []messageCompletionChoice{
			{
				Message: messageCompletionChoiceMessage{
					Role:    "assistant",
					Content: text.String(),
				},
				FinishReason: finishReason,
			},
		},
		// The prompt counts every input token, the ones read from or written to the cache included.
		Usage: messageCompletionUsage{
			PromptTokens:     message.Usage.InputTokens + message.Usage.CacheReadInputTokens + message.Usage.CacheCreationInputTokens,
			CompletionTokens: message.Usage.OutputTokens,
			TotalTokens:      message.Usage.InputTokens + message.Usage.CacheReadInputTokens + message.Usage.CacheCreationInputTokens + message.Usage.OutputTokens,
		},
	}

	body, err := json.Marshal(completion)
	if err != nil {
		return proxias.CompletionResponse{}, fmt.Errorf("anthropic: %w", err)
	}
	return proxias.CompletionResponse{Status: http.StatusOK, ContentType: "application/json", Body: io.NopCloser(bytes.NewReader(body))}, nil
}

type messageCompletion struct {
	ID      string                    `json:"id"`
	Object  string                    `json:"object"`
	Created int64                     `json:"created"`
	Model   string                    `json:"model"`
	Choices []messageCompletionChoice `json:"choices"`
	Usage   messageCompletionUsage    `json:"usage"`
}

type messageCompletionChoice struct {
	Index        int                            `json:"index"`
	Message      messageCompletionChoiceMessage `json:"message"`
	FinishReason string                         `json:"finish_reason"`
}

type messageCompletionChoiceMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type messageCompletionUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}
