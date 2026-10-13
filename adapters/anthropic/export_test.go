package anthropic

import (
	"context"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// This file exports to the tests what the package keeps unexported.

// ClaudeHelper is claudeHelper, for the tests.
type ClaudeHelper = claudeHelper

// NewClaudeHelper makes a claudeHelper, for the tests.
func NewClaudeHelper(client anthropicsdk.Client, model anthropicsdk.Model) ClaudeHelper {
	return claudeHelper{client, model}
}

// Ask is ask, for the tests.
func (c claudeHelper) Ask(ctx context.Context, instructions string, input any) (*anthropicsdk.Message, error) {
	return c.ask(ctx, instructions, input)
}

// AskJSON is askJSON, for the tests.
func (c claudeHelper) AskJSON(ctx context.Context, instructions string, input, answer any) error {
	return c.askJSON(ctx, instructions, input, answer)
}
