// Package openai adapts the OpenAI chat completions API (POST /v1/chat/completions) to a proxias.ControlPlane:
// it decodes a request into a proxias.CompletionRequest, and Handler serves a ControlPlane over HTTP.
//
// Transports are the host's, which can use this package as is, or write its own handler around
// DecodeRequest.
package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/morebec/proxias"
)

// ChatCompletionRequest is the body of a chat completions request, reduced to what Proxias reads of it.
type ChatCompletionRequest struct {
	Model    string                  `json:"model"`
	Stream   bool                    `json:"stream"`
	Messages []ChatCompletionMessage `json:"messages"`
}

// ChatCompletionMessage is a message of a ChatCompletionRequest. Its Content is a JSON string, or a list of
// content parts, or null (an assistant message that only calls tools).
type ChatCompletionMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// DecodeRequest reads the body of a chat completions request.
//
// The proxias.CompletionRequest keeps the body byte for byte in Request, so that it can be forwarded as is
// to an upstream model. The content of a message becomes plain text: the string itself, or the text parts
// of a list joined by new lines (the other parts are left out), or nothing for null.
//
// An error is returned if the body is not a JSON chat completions request, has no messages, or has a
// message whose content is neither a string, a list nor null.
func DecodeRequest(r io.Reader) (proxias.CompletionRequest, error) {
	body, err := io.ReadAll(r)
	if err != nil {
		return proxias.CompletionRequest{}, fmt.Errorf("openai: read the request: %w", err)
	}

	var wire ChatCompletionRequest
	if err := json.Unmarshal(body, &wire); err != nil {
		return proxias.CompletionRequest{}, fmt.Errorf("openai: decode the request: %w", err)
	}
	if len(wire.Messages) == 0 {
		return proxias.CompletionRequest{}, errors.New("openai: the request has no messages")
	}

	messages := make([]proxias.CompletionMessage, len(wire.Messages))
	for i, message := range wire.Messages {
		content, err := contentText(message.Content)
		if err != nil {
			return proxias.CompletionRequest{}, fmt.Errorf("openai: message %d: %w", i, err)
		}
		messages[i] = proxias.CompletionMessage{Role: proxias.CompletionMessageRole(message.Role), Content: content}
	}

	return proxias.CompletionRequest{
		Stream:   wire.Stream,
		Model:    wire.Model,
		Messages: messages,
		Request:  body,
	}, nil
}

func contentText(raw json.RawMessage) ([]byte, error) {
	switch {
	case len(raw) == 0 || string(raw) == "null":
		return nil, nil
	case raw[0] == '"':
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, fmt.Errorf("decode the content: %w", err)
		}
		return []byte(text), nil
	case raw[0] == '[':
		var parts []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(raw, &parts); err != nil {
			return nil, fmt.Errorf("decode the content parts: %w", err)
		}
		var texts []string
		for _, part := range parts {
			if part.Type == "text" {
				texts = append(texts, part.Text)
			}
		}
		return []byte(strings.Join(texts, "\n")), nil
	}

	return nil, errors.New("unsupported content: expected a string, a list of parts or null")
}
