package anthropic_test

import (
	"context"
	"net/http"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk/adapters/anthropic"
)

func TestUsageCounter(t *testing.T) {
	// Given
	var counter anthropic.UsageCounter
	var request map[string]any
	client := mockClaude(t, http.StatusOK, `{"id":"m","type":"message","role":"assistant","model":"claude-haiku-5-5",
		"content":[{"type":"text","text":"Hi."}],"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":3,"cache_creation_input_tokens":5,"cache_read_input_tokens":7}}`, &request)
	counted := anthropicsdk.NewClient(append(client.Options, option.WithMiddleware(counter.Middleware()))...)

	// When
	for range 2 {
		message, err := counted.Messages.New(context.Background(), anthropicsdk.MessageNewParams{Model: "claude-haiku-5-5", MaxTokens: 16, Messages: []anthropicsdk.MessageParam{anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi"))}})
		require.NoError(t, err)
		assert.Equal(t, "Hi.", message.Content[0].Text, "the response is read whole after it is counted")
	}
	usage := counter.Reset()

	// Then
	assert.Positive(t, usage.CallDuration, "the calls are timed")
	usage.CallDuration = 0
	assert.Equal(t, anthropic.Usage{Calls: 2, InputTokens: 20, OutputTokens: 6, CacheCreationInputTokens: 10, CacheReadInputTokens: 14}, usage)
	assert.Equal(t, anthropic.Usage{}, counter.Usage(), "Reset counts from zero again")
}
