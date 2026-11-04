package anthropic_test

import (
	"context"
	"net/http"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
)

func TestUsageCounter(t *testing.T) {
	// Given
	var counter anthropic.UsageCounter
	var request map[string]any
	client := mockClaude(
		t,
		http.StatusOK,
		`{"id":"m","type":"message","role":"assistant","model":"claude-haiku-5-5",
		"content":[{"type":"text","text":"Hi."}],"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":3,"cache_creation_input_tokens":5,"cache_read_input_tokens":7}}`,
		&request,
	)
	counted := anthropicsdk.NewClient(
		append(client.Options, option.WithMiddleware(counter.Middleware()))...)

	// When
	for range 2 {
		message, err := counted.Messages.New(
			context.Background(),
			anthropicsdk.MessageNewParams{
				Model:     "claude-haiku-5-5",
				MaxTokens: 16,
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
				},
			},
		)
		require.NoError(t, err)
		assert.Equal(t, "Hi.", message.Content[0].Text, "the response is read whole after it is counted")
	}
	usage := counter.Reset()

	// Then
	assert.Positive(t, usage.CallDuration, "the calls are timed")
	usage.CallDuration = 0
	assert.Equal(
		t,
		anthropic.Usage{
			Calls:                    2,
			InputTokens:              20,
			OutputTokens:             6,
			CacheCreationInputTokens: 10,
			CacheReadInputTokens:     14,
		},
		usage,
	)
	assert.Equal(
		t,
		anthropic.Usage{},
		counter.Usage(),
		"Reset counts from zero again",
	)
}

func TestUsageCounter_Streamed(t *testing.T) {
	// Given
	var counter anthropic.UsageCounter
	var request map[string]any
	client := mockStreamingClaude(
		t,
		"event: message_start\n"+
			`data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":10,"output_tokens":1,"cache_creation_input_tokens":5,"cache_read_input_tokens":7}}}`+"\n\n"+
			"event: content_block_start\n"+
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n"+
			"event: content_block_delta\n"+
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi."}}`+"\n\n"+
			"event: content_block_stop\n"+
			`data: {"type":"content_block_stop","index":0}`+"\n\n"+
			"event: message_delta\n"+
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`+"\n\n"+
			"event: message_stop\n"+
			`data: {"type":"message_stop"}`+"\n\n",
		&request,
	)
	counted := anthropicsdk.NewClient(
		append(client.Options, option.WithMiddleware(counter.Middleware()))...)
	llm := anthropic.NewLLM(counted, "m")

	// When
	var text string
	for chunk, err := range llm.Complete(context.Background(), proxias.LLMRequest{}) {
		require.NoError(t, err)
		text += chunk.Text
	}
	usage := counter.Reset()

	// Then
	assert.Equal(t, "Hi.", text, "the stream is read whole after it is counted")
	usage.CallDuration = 0
	assert.Equal(
		t,
		anthropic.Usage{
			Calls:                    1,
			InputTokens:              10,
			OutputTokens:             3,
			CacheCreationInputTokens: 5,
			CacheReadInputTokens:     7,
		},
		usage,
	)
}
