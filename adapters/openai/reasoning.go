// Package openai implements the components of Proxias with a model served by an OpenAI-compatible chat
// completions API: OpenAI itself, or any provider or server that speaks it. It goes through the official OpenAI
// SDK.
//
// The host builds the SDK client, and so owns the API key, the base URL of the provider, the timeout and the
// retries, and chooses the model. A failing call of a component returns the SDK's error, wrapped: errors.As finds
// an *openai.Error of the SDK, and its status code. The ReasoningEngine is a transparent proxy: it returns the
// provider's error responses as they are.
package openai

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	openaisdk "github.com/openai/openai-go/v3"

	"github.com/morebec/proxias"
)

// ReasoningEngine is a proxias.ReasoningEngine that is a transparent proxy to the model: Proxias's model is the
// chat completions API, so nothing is mapped. The request keeps every field that the harness sent, and the
// provider's response comes back as it was sent: its status, its headers, its body, a stream or an error.
type ReasoningEngine struct {
	openaiHelper
}

// NewReasoningEngine makes a ReasoningEngine that asks model through client. An empty model keeps the one that
// the harness asked for.
func NewReasoningEngine(client openaisdk.Client, model string) *ReasoningEngine {
	return &ReasoningEngine{openaiHelper{client, model}}
}

// Reason implements proxias.ReasoningEngine. The workflows are told to the model in a system message, after the
// harness's own ones. An error response of the provider is returned as a response, as it came: only a call that
// got no response is an error.
func (e *ReasoningEngine) Reason(ctx context.Context, req proxias.ChatCompletionRequest, descs []proxias.WorkflowDescriptor) (proxias.ChatCompletionResponse, error) {
	if e.model != "" {
		req.Model = e.model
	}
	if len(descs) != 0 {
		req.Messages = withSystemMessage(req.Messages, workflowsMessage(descs))
	}

	// Into an *http.Response, the SDK hands the response over unread, so that a stream is forwarded as it comes.
	var resp *http.Response
	if err := e.client.Post(ctx, "chat/completions", req, &resp); err != nil {
		var apiErr *openaisdk.Error
		if !errors.As(err, &apiErr) || apiErr.Response == nil {
			return proxias.ChatCompletionResponse{}, fmt.Errorf("openai: %w", err)
		}
		resp = apiErr.Response // its body was read into the error, and put back
	}

	// The hop-by-hop headers are of the connection to the provider, not of its response.
	header := resp.Header.Clone()
	for _, name := range []string{"Connection", "Proxy-Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization", "Te", "Trailer", "Transfer-Encoding", "Upgrade"} {
		header.Del(name)
	}
	return proxias.ChatCompletionResponse{Status: resp.StatusCode, Header: header, Body: resp.Body}, nil
}

// workflowsMessage tells the model what the application can do by running workflows, so that it answers a
// question about its capabilities accurately. The workflows are listed by name and description: their schemas are
// for the classifier, not for the reasoning.
func workflowsMessage(descs []proxias.WorkflowDescriptor) string {
	prompt := proxias.NewPrompt().AddSection("Workflows",
		"The application also runs these workflows for the requests that ask for them. When asked what the application can do, describe them:")
	for _, d := range descs {
		prompt.AddLabeledBulletPoint(d.Name, d.Description)
	}
	return prompt.String()
}

// withSystemMessage is messages with a system message of text after the leading system and developer ones, so
// that the prefix of the harness's prompt stays cached. It makes a new slice: the caller's is shared.
func withSystemMessage(messages []proxias.ChatMessage, text string) []proxias.ChatMessage {
	at := 0
	for at < len(messages) && (messages[at].Role == proxias.ChatMessageRoleSystem || messages[at].Role == proxias.ChatMessageRoleDeveloper) {
		at++
	}

	out := make([]proxias.ChatMessage, 0, len(messages)+1)
	out = append(out, messages[:at]...)
	out = append(out, proxias.ChatMessage{Role: proxias.ChatMessageRoleSystem, Content: textContent(text)})
	return append(out, messages[at:]...)
}
