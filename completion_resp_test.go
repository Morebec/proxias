package proxias_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func TestNewChatCompletionResponse(t *testing.T) {
	// Given
	completion := proxias.ChatCompletion{
		ID: "c1", Object: "chat.completion", Model: "m",
		Choices: []proxias.ChatCompletionChoice{{
			Message:      proxias.ChatMessage{Role: proxias.ChatMessageRoleAssistant, Content: proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: "hi"}}},
			FinishReason: proxias.ChatFinishReasonStop,
		}},
	}

	// When
	got, err := proxias.NewChatCompletionResponse(completion)

	// Then
	require.NoError(t, err)
	defer func() { _ = got.Body.Close() }()
	assert.Equal(t, http.StatusOK, got.Status)
	assert.Equal(t, "application/json", got.Header.Get("Content-Type"))
	raw, err := io.ReadAll(got.Body)
	require.NoError(t, err)
	assert.JSONEq(t, `{"id":"c1","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`, string(raw))
}

func TestNewChatCompletionStream(t *testing.T) {
	// Given
	stop := proxias.ChatFinishReasonStop
	produce := func(emit func(proxias.ChatCompletionChunk) error) error {
		for _, chunk := range []proxias.ChatCompletionChunk{
			{ID: "c1", Choices: []proxias.ChatCompletionChunkChoice{{Delta: proxias.ChatCompletionDelta{Role: "assistant", Content: "hi"}}}},
			{ID: "c1", Choices: []proxias.ChatCompletionChunkChoice{{FinishReason: &stop}}},
		} {
			if err := emit(chunk); err != nil {
				return err
			}
		}
		return nil
	}

	// When
	got := proxias.NewChatCompletionStream(produce)

	// Then
	defer func() { _ = got.Body.Close() }()
	assert.Equal(t, http.StatusOK, got.Status)
	assert.Equal(t, "text/event-stream", got.Header.Get("Content-Type"))
	raw, err := io.ReadAll(got.Body)
	require.NoError(t, err)
	events := strings.Split(strings.TrimSpace(string(raw)), "\n\n")
	require.Len(t, events, 3)
	assert.Equal(t, `data: {"id":"c1","object":"","created":0,"model":"","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}`, events[0])
	assert.Contains(t, events[1], `"delta":{},"finish_reason":"stop"`)
	assert.Equal(t, "data: [DONE]", events[2])
}

func TestNewChatCompletionStreamFailsWhenTheProducerFails(t *testing.T) {
	// Given
	broken := errors.New("the provider went away")

	// When
	got := proxias.NewChatCompletionStream(func(func(proxias.ChatCompletionChunk) error) error { return broken })

	// Then
	defer func() { _ = got.Body.Close() }()
	raw, err := io.ReadAll(got.Body)
	assert.ErrorIs(t, err, broken)
	assert.NotContains(t, string(raw), "[DONE]")
}

func TestNewChatCompletionStreamStopsTheProducerWhenTheBodyIsClosed(t *testing.T) {
	// Given
	stopped := make(chan error, 1)
	got := proxias.NewChatCompletionStream(func(emit func(proxias.ChatCompletionChunk) error) error {
		for {
			if err := emit(proxias.ChatCompletionChunk{}); err != nil {
				stopped <- err
				return err
			}
		}
	})

	// When the client goes away
	require.NoError(t, got.Body.Close())

	// Then
	select {
	case err := <-stopped:
		assert.Error(t, err)
	case <-time.After(time.Second):
		t.Fatal("the producer was not stopped")
	}
}
