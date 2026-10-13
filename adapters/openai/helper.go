package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	openaisdk "github.com/openai/openai-go/v3"

	"github.com/morebec/proxias"
)

// openaiHelper is what the components that ask the model embed: the SDK client, the model, and the one call they
// make. The decisions (the prompts, the answers they read, what they keep of them) are each component's own.
type openaiHelper struct {
	client openaisdk.Client
	model  string
}

// ask asks the model to follow the instructions for input, which is encoded as JSON in the only user turn, and
// returns its answer. The request and the answer are Proxias's own model, which is the API's.
func (h openaiHelper) ask(ctx context.Context, instructions string, input any) (proxias.ChatCompletion, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return proxias.ChatCompletion{}, fmt.Errorf("encode the input: %w", err)
	}

	req := proxias.ChatCompletionRequest{
		Model: h.model,
		Messages: []proxias.ChatMessage{
			{Role: proxias.ChatMessageRoleSystem, Content: textContent(instructions)},
			{Role: proxias.ChatMessageRoleUser, Content: textContent(string(data))},
		},
	}
	var completion proxias.ChatCompletion
	if err := h.client.Post(ctx, "chat/completions", req, &completion); err != nil {
		return proxias.ChatCompletion{}, err
	}
	return completion, nil
}

// askJSON is ask for an answer that is the JSON object of answer.
func (h openaiHelper) askJSON(ctx context.Context, instructions string, input, answer any) error {
	completion, err := h.ask(ctx, instructions, input)
	if err != nil {
		return err
	}
	return decodeJSONObject(completionText(completion), answer)
}

// completionText is the text that the model wrote, in the first answer of the completion: the one asked for.
func completionText(completion proxias.ChatCompletion) string {
	if len(completion.Choices) == 0 {
		return ""
	}
	return completion.Choices[0].Message.Content.Text()
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

func textContent(text string) proxias.ChatContent {
	return proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: text}}
}
