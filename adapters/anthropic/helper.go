package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// claudeHelper is what the components that ask the model embed: the SDK client, the model, and the one call they
// make. The decisions (the prompts, the answers they read, what they keep of them) are each component's own.
type claudeHelper struct {
	client anthropicsdk.Client
	model  anthropicsdk.Model
}

// ask asks the model to follow the instructions for input, which is encoded as JSON in the only user turn, and
// returns its answer. The instructions never vary, so the API may cache them.
func (c claudeHelper) ask(ctx context.Context, instructions string, input any) (*anthropicsdk.Message, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("encode the input: %w", err)
	}

	return c.client.Messages.New(ctx, anthropicsdk.MessageNewParams{
		Model:     c.model,
		MaxTokens: defaultMaxTokens,
		System: []anthropicsdk.TextBlockParam{{
			Text:         instructions,
			CacheControl: anthropicsdk.NewCacheControlEphemeralParam(),
		}},
		Messages: []anthropicsdk.MessageParam{anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock(string(data)))},
	})
}

// askJSON is ask for an answer that is the JSON object of answer.
func (c claudeHelper) askJSON(ctx context.Context, instructions string, input, answer any) error {
	message, err := c.ask(ctx, instructions, input)
	if err != nil {
		return err
	}
	return decodeJSONObject(messageToString(message), answer)
}

// messageToString is the text that the model wrote.
func messageToString(message *anthropicsdk.Message) string {
	var text strings.Builder
	for _, block := range message.Content {
		if t, ok := block.AsAny().(anthropicsdk.TextBlock); ok {
			text.WriteString(t.Text)
		}
	}
	return text.String()
}

// decodeJSONObject reads the JSON object of the model's text into v. The model is told to answer with the
// object only: a code fence or words around it are tolerated.
func decodeJSONObject(text string, v any) error {
	start, end := strings.Index(text, "{"), strings.LastIndex(text, "}")
	if start < 0 || end < start {
		return errors.New("the model's answer has no JSON object")
	}
	if err := json.Unmarshal([]byte(text[start:end+1]), v); err != nil {
		return fmt.Errorf("the model's answer is not the expected JSON: %w", err)
	}
	return nil
}
