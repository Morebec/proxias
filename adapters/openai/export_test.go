package openai

import (
	"context"

	openaisdk "github.com/openai/openai-go/v3"

	"github.com/morebec/proxias"
)

// This file exports to the tests what the package keeps unexported.

// OpenAIHelper is openaiHelper, for the tests.
type OpenAIHelper = openaiHelper

// NewOpenAIHelper makes an openaiHelper, for the tests.
func NewOpenAIHelper(client openaisdk.Client, model string) OpenAIHelper {
	return openaiHelper{client, model}
}

// Ask is ask, for the tests.
func (h openaiHelper) Ask(ctx context.Context, instructions string, input any) (proxias.ChatCompletion, error) {
	return h.ask(ctx, instructions, input)
}

// AskJSON is askJSON, for the tests.
func (h openaiHelper) AskJSON(ctx context.Context, instructions string, input, answer any) error {
	return h.askJSON(ctx, instructions, input, answer)
}
