package anthropic_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
)

const someModel = "some-model"

// answer is the API's answer to a message: two blocks of text, and a cache read.
const answer = `{"id":"msg_1","model":"claude-x","stop_reason":"end_turn","content":[{"type":"text","text":"Hi "},{"type":"text","text":"there"}],"usage":{"input_tokens":10,"cache_read_input_tokens":5,"output_tokens":4}}`

// mockAnthropic is a server that answers every call with status and body, and keeps the body of the last
// request it got.
func mockAnthropic(t *testing.T, status int, body string) (anthropicsdk.Client, *map[string]any) {
	t.Helper()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &got)

		w.Header().Set("content-type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)

	client := anthropicsdk.NewClient(option.WithAPIKey("secret-key"), option.WithBaseURL(server.URL), option.WithMaxRetries(0))
	return client, &got
}

func message(role proxias.CompletionMessageRole, content string) proxias.CompletionMessage {
	return proxias.CompletionMessage{Role: role, Content: []byte(content)}
}

func TestReasoningEngineReason(t *testing.T) {
	type expect struct {
		system   []any
		messages []any
	}
	tests := []struct {
		name  string
		given []proxias.CompletionMessage
		then  expect
	}{
		{
			name: "canonical",
			given: []proxias.CompletionMessage{
				message(proxias.CompletionMessageRoleUser, "hello"),
			},
			then: expect{
				messages: []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "hello"}}}},
			},
		},
		{
			name: "GIVEN system and developer messages THEN they are the system prompt",
			given: []proxias.CompletionMessage{
				message(proxias.CompletionMessageRoleSystem, "We run a hosting company"),
				message(proxias.CompletionMessageRoleDeveloper, "Be brief"),
				message(proxias.CompletionMessageRoleUser, "hello"),
			},
			then: expect{
				system: []any{
					map[string]any{"type": "text", "text": "We run a hosting company"},
					map[string]any{"type": "text", "text": "Be brief"},
				},
				messages: []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "hello"}}}},
			},
		},
		{
			name: "GIVEN a conversation THEN the roles are kept, and empty or tool messages are left out",
			given: []proxias.CompletionMessage{
				message(proxias.CompletionMessageRoleUser, "hello"),
				message(proxias.CompletionMessageRoleAssistant, "hi"),
				message(proxias.CompletionMessageRoleTool, "some tool output"),
				message(proxias.CompletionMessageRoleUser, " "),
				message(proxias.CompletionMessageRoleUser, "how are you?"),
			},
			then: expect{
				messages: []any{
					map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "hello"}}},
					map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "hi"}}},
					map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "how are you?"}}},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, sent := mockAnthropic(t, 200, answer)
			engine := anthropic.NewReasoningEngine(client, someModel)

			got, err := engine.Reason(t.Context(), proxias.CompletionRequest{Messages: tt.given}, nil)

			require.NoError(t, err)
			assert.Equal(t, http.StatusOK, got.Status)
			assert.Equal(t, "application/json", got.ContentType)
			var completion struct {
				ID      string `json:"id"`
				Object  string `json:"object"`
				Model   string `json:"model"`
				Choices []struct {
					Message struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					} `json:"message"`
					FinishReason string `json:"finish_reason"`
				} `json:"choices"`
				Usage struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
					TotalTokens      int `json:"total_tokens"`
				} `json:"usage"`
			}
			require.NoError(t, json.NewDecoder(got.Body).Decode(&completion))
			assert.Equal(t, "msg_1", completion.ID)
			assert.Equal(t, "chat.completion", completion.Object)
			assert.Equal(t, "claude-x", completion.Model)
			require.Len(t, completion.Choices, 1)
			assert.Equal(t, "assistant", completion.Choices[0].Message.Role)
			assert.Equal(t, "Hi there", completion.Choices[0].Message.Content, "the text blocks are joined")
			assert.Equal(t, "stop", completion.Choices[0].FinishReason)
			assert.Equal(t, 15, completion.Usage.PromptTokens, "the cached input tokens count")
			assert.Equal(t, 4, completion.Usage.CompletionTokens)
			assert.Equal(t, 19, completion.Usage.TotalTokens)
			assert.Equal(t, someModel, (*sent)["model"])
			assert.Equal(t, tt.then.messages, (*sent)["messages"])
			if tt.then.system == nil {
				assert.NotContains(t, *sent, "system")
			} else {
				assert.Equal(t, tt.then.system, (*sent)["system"])
			}
		})
	}
}

func TestReasoningEngineReasonReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockAnthropic(t, 429, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
	engine := anthropic.NewReasoningEngine(client, someModel)

	// When
	got, err := engine.Reason(t.Context(), proxias.CompletionRequest{Messages: []proxias.CompletionMessage{message(proxias.CompletionMessageRoleUser, "hello")}}, nil)

	// Then
	var apiErr *anthropicsdk.Error
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, 429, apiErr.StatusCode)
	assert.Contains(t, err.Error(), "anthropic: ")
	assert.Nil(t, got.Body)
}
