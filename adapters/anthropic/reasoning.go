// Package anthropic implements the components of Proxias with Claude, through the official Anthropic SDK.
//
// The host builds the SDK client, and so owns the API key, the timeout, the retries and the base URL, and
// always chooses the model. A failing call returns the SDK's error, wrapped: errors.As finds an
// *anthropic.Error of the SDK, and its status code.
package anthropic

import (
	"context"
	"fmt"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/morebec/proxias"
)

const defaultMaxTokens = 4096

// ReasoningEngine is a proxias.ReasoningEngine that has Claude answer the conversation as it is. It answers
// as the OpenAI chat completions API does, the transport of the openai package.
type ReasoningEngine struct {
	client anthropicsdk.Client
	model  anthropicsdk.Model
}

// NewReasoningEngine makes a ReasoningEngine that asks model through client.
func NewReasoningEngine(client anthropicsdk.Client, model anthropicsdk.Model) *ReasoningEngine {
	return &ReasoningEngine{client: client, model: model}
}

// Reason implements proxias.ReasoningEngine. System and developer messages are Claude's system prompt, user
// and assistant messages its conversation; messages with other roles, or without content, are left out.
// The workflow descriptors are not used.
//
// The answer is a chat completion. req.Stream is ignored: the answer is never streamed.
func (e *ReasoningEngine) Reason(ctx context.Context, req proxias.CompletionRequest, descriptors []proxias.WorkflowDescriptor) (proxias.CompletionResponse, error) {
	params := anthropicsdk.MessageNewParams{Model: e.model, MaxTokens: defaultMaxTokens}

	for _, m := range req.Messages {
		content := string(m.Content)
		if strings.TrimSpace(content) == "" {
			continue
		}

		switch m.Role {
		case proxias.CompletionMessageRoleSystem, proxias.CompletionMessageRoleDeveloper:
			params.System = append(params.System, anthropicsdk.TextBlockParam{Text: content})
		case proxias.CompletionMessageRoleUser:
			params.Messages = append(params.Messages, anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock(content)))
		case proxias.CompletionMessageRoleAssistant:
			params.Messages = append(params.Messages, anthropicsdk.NewAssistantMessage(anthropicsdk.NewTextBlock(content)))
		}
	}

	response, err := e.client.Messages.New(ctx, params)
	if err != nil {
		return proxias.CompletionResponse{}, fmt.Errorf("anthropic: %w", err)
	}

	return mapAnthropicMessageToCompletionResponse(response)
}
