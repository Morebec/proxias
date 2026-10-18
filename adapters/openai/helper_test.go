package openai_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

const someModel = "some-model"

var (
	restartService = proxias.ActionDescriptor{
		ID:          "restart_service",
		Name:        "Restart a service",
		Description: "Restarts a service",
		InputSchema: map[string]any{"type": "object", "required": []any{"service"}},
	}
	appVersion = proxias.ActionDescriptor{ID: "app_version", Name: "App version", Description: "Tells the running version"}
)

// sent is what the mock received: the path and the body of the last request.
type sent struct {
	path string
	body string
}

// mockOpenAI is a server that answers every call with status and body, as a stream if the body is one, and keeps
// the last request it got.
func mockOpenAI(t *testing.T, status int, body string) (openaisdk.Client, *sent) {
	t.Helper()

	var got sent
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		got = sent{path: r.URL.Path, body: string(raw)}

		contentType := "application/json"
		if strings.HasPrefix(body, "data:") {
			contentType = "text/event-stream"
		}
		w.Header().Set("Content-Type", contentType)
		w.Header().Set("X-Request-Id", "req_1")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)

	client := openaisdk.NewClient(option.WithAPIKey("secret-key"), option.WithBaseURL(server.URL), option.WithMaxRetries(0))
	return client, &got
}

// answerWith is the API's answer, in which the model wrote text.
func answerWith(text string) string {
	content, _ := json.Marshal(text)
	return `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":` + string(content) + `},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`
}

func text(s string) proxias.ChatContent {
	return proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: s}}
}

func userSays(s string) proxias.ChatCompletionRequest {
	return proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{{Role: proxias.ChatMessageRoleUser, Content: text(s)}}}
}

func TestOpenAIHelperAsk(t *testing.T) {
	// Given
	client, got := mockOpenAI(t, http.StatusOK, answerWith("hi"))
	helper := openai.NewOpenAIHelper(client, someModel)

	// When
	completion, err := helper.Ask(t.Context(), "follow these", map[string]int{"a": 1})

	// Then the instructions are the system message, the input the JSON of the only user message, and the answer
	// the model's completion
	require.NoError(t, err)
	assert.Equal(t, "chatcmpl-1", completion.ID)
	assert.Equal(t, "hi", completion.Choices[0].Message.Content.Text())
	assert.Equal(t, "/chat/completions", got.path)
	assert.JSONEq(t, `{"model":"some-model","messages":[{"role":"system","content":"follow these"},{"role":"user","content":"{\"a\":1}"}]}`, got.body)
}

func TestOpenAIHelperAskFailsForAnInputThatIsNotJSON(t *testing.T) {
	// Given
	client, got := mockOpenAI(t, http.StatusOK, answerWith("hi"))
	helper := openai.NewOpenAIHelper(client, someModel)

	// When
	_, err := helper.Ask(t.Context(), "follow these", make(chan int))

	// Then
	assert.ErrorContains(t, err, "encode the input")
	assert.Empty(t, got.body, "the API was not called")
}

func TestOpenAIHelperAskJSON(t *testing.T) {
	type answer struct {
		Value string `json:"value"`
	}
	tests := []struct {
		name  string
		given string // the API's answer
		then  assert.ErrorAssertionFunc
	}{
		{
			name:  "canonical",
			given: answerWith(`{"value":"x"}`),
			then:  assert.NoError,
		},
		{
			name:  "GIVEN the JSON is in a code fence with words around it THEN it is read",
			given: answerWith("Sure:\n```json\n{\"value\":\"x\"}\n```"),
			then:  assert.NoError,
		},
		{
			name:  "GIVEN an answer that is not JSON THEN it fails",
			given: answerWith("x"),
			then:  assert.Error,
		},
		{
			name:  "GIVEN a JSON of the wrong shape THEN it fails",
			given: answerWith(`{"value":1}`),
			then:  assert.Error,
		},
		{
			name:  "GIVEN an answer without a choice THEN it fails",
			given: `{"id":"chatcmpl-1","object":"chat.completion","choices":[]}`,
			then:  assert.Error,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, _ := mockOpenAI(t, http.StatusOK, tt.given)
			helper := openai.NewOpenAIHelper(client, someModel)
			var got answer

			err := helper.AskJSON(t.Context(), "follow these", struct{}{}, &got)

			tt.then(t, err)
			if err == nil {
				assert.Equal(t, answer{Value: "x"}, got)
			}
		})
	}
}
