package proxias_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

func TestChatCompletionBuilderBuildText(t *testing.T) {
	// Given
	builder := proxias.NewChatCompletionBuilder(proxias.ChatCompletionRequest{Model: "m"})

	// When
	got := builder.BuildText("hi")

	// Then
	assert.NotEmpty(t, got.ID)
	assert.Equal(t, "chat.completion", got.Object)
	assert.Equal(t, "m", got.Model)
	require.Len(t, got.Choices, 1)
	assert.Equal(t, proxias.ChatMessageRoleAssistant, got.Choices[0].Message.Role)
	assert.Equal(t, "hi", got.Choices[0].Message.Content.Text())
	assert.Equal(t, proxias.ChatFinishReasonStop, got.Choices[0].FinishReason)
}

func TestChatCompletionBuilderBuildToolCalls(t *testing.T) {
	// Given
	builder := proxias.NewChatCompletionBuilder(proxias.ChatCompletionRequest{Model: "m"})
	call := proxias.ChatToolCall{ID: "call1", Type: "function", Function: proxias.ChatFunctionCall{Name: "ask_user", Arguments: "{}"}}

	// When
	got := builder.BuildToolCalls(call)

	// Then
	assert.Equal(t, "m", got.Model)
	require.Len(t, got.Choices, 1)
	assert.Equal(t, proxias.ChatMessageRoleAssistant, got.Choices[0].Message.Role)
	assert.Equal(t, []proxias.ChatToolCall{call}, got.Choices[0].Message.ToolCalls)
	assert.Nil(t, got.Choices[0].Message.Content)
	assert.Equal(t, proxias.ChatFinishReasonToolCalls, got.Choices[0].FinishReason)
}
