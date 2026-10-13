package openai_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

func TestReasoningEngineReasonForwardsTheRequestAsItIs(t *testing.T) {
	tests := []struct {
		name  string
		given string // the model of the engine
		then  string // the request sent
	}{
		{
			name:  "canonical",
			given: someModel,
			then:  `{"model":"some-model","seed":7,"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name:  "GIVEN no model THEN the harness's is kept",
			given: "",
			then:  `{"model":"harness-model","seed":7,"messages":[{"role":"user","content":"hi"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, got := mockOpenAI(t, http.StatusOK, answerWith("Hi"))
			engine := openai.NewReasoningEngine(client, tt.given)
			var req proxias.ChatCompletionRequest
			require.NoError(t, json.Unmarshal([]byte(`{"model":"harness-model","seed":7,"messages":[{"role":"user","content":"hi"}]}`), &req))

			resp, err := engine.Reason(t.Context(), req, nil)

			require.NoError(t, err)
			_ = resp.Body.Close()
			assert.Equal(t, "/chat/completions", got.path)
			assert.JSONEq(t, tt.then, got.body)
		})
	}
}

func TestReasoningEngineReasonReturnsTheProvidersResponseAsItIs(t *testing.T) {
	stream := "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n"
	refusal := `{"error":{"message":"bad model","type":"invalid_request_error"}}`
	tests := []struct {
		name   string
		status int
		body   string
	}{
		{
			name:   "canonical",
			status: http.StatusOK,
			body:   answerWith("Hi"),
		},
		{
			name:   "GIVEN a stream THEN it is forwarded as it comes",
			status: http.StatusOK,
			body:   stream,
		},
		{
			name:   "GIVEN an error response THEN it is forwarded, not made an error",
			status: http.StatusBadRequest,
			body:   refusal,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, _ := mockOpenAI(t, tt.status, tt.body)
			engine := openai.NewReasoningEngine(client, someModel)

			resp, err := engine.Reason(t.Context(), proxias.ChatCompletionRequest{}, nil)

			require.NoError(t, err)
			defer func() { _ = resp.Body.Close() }()
			assert.Equal(t, tt.status, resp.Status)
			assert.Equal(t, "req_1", resp.Header.Get("X-Request-Id"), "the provider's headers are forwarded")
			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			assert.Equal(t, tt.body, string(body))
		})
	}
}

func TestReasoningEngineReasonDoesNotForwardTheHeadersOfTheConnection(t *testing.T) {
	// Given a provider whose response has a hop-by-hop header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Keep-Alive", "timeout=5")
		_, _ = w.Write([]byte(answerWith("Hi")))
	}))
	t.Cleanup(server.Close)
	client := openaisdk.NewClient(option.WithAPIKey("secret-key"), option.WithBaseURL(server.URL), option.WithMaxRetries(0))

	// When
	resp, err := openai.NewReasoningEngine(client, someModel).Reason(t.Context(), proxias.ChatCompletionRequest{}, nil)

	// Then
	require.NoError(t, err)
	_ = resp.Body.Close()
	assert.Equal(t, "application/json", resp.Header.Get("Content-Type"))
	assert.Empty(t, resp.Header.Get("Keep-Alive"))
}

func TestReasoningEngineReasonFailsWhenTheProviderCannotBeReached(t *testing.T) {
	// Given a provider that is gone
	server := httptest.NewServer(http.NotFoundHandler())
	server.Close()
	client := openaisdk.NewClient(option.WithAPIKey("secret-key"), option.WithBaseURL(server.URL), option.WithMaxRetries(0))

	// When
	_, err := openai.NewReasoningEngine(client, someModel).Reason(t.Context(), proxias.ChatCompletionRequest{}, nil)

	// Then there is no response to forward: it is an error
	assert.ErrorContains(t, err, "openai: ")
}

func TestReasoningEngineReasonTellsTheModelTheWorkflows(t *testing.T) {
	system := proxias.ChatMessage{Role: proxias.ChatMessageRoleSystem, Content: text("be brief")}
	user := proxias.ChatMessage{Role: proxias.ChatMessageRoleUser, Content: text("what can you do?")}
	tests := []struct {
		name  string
		given []proxias.WorkflowDescriptor
		then  []string // the roles of the messages sent
	}{
		{
			name:  "GIVEN workflows THEN a system message tells them, after the harness's own",
			given: []proxias.WorkflowDescriptor{restartService},
			then:  []string{"system", "system", "user"},
		},
		{
			name:  "GIVEN no workflow THEN no message is added",
			given: nil,
			then:  []string{"system", "user"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, got := mockOpenAI(t, http.StatusOK, answerWith("ok"))
			engine := openai.NewReasoningEngine(client, someModel)
			req := proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{system, user}}

			resp, err := engine.Reason(t.Context(), req, tt.given)

			require.NoError(t, err)
			_ = resp.Body.Close()
			var body struct {
				Messages []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal([]byte(got.body), &body))
			roles := make([]string, 0, len(body.Messages))
			for _, m := range body.Messages {
				roles = append(roles, m.Role)
			}
			assert.Equal(t, tt.then, roles)
			if len(tt.given) > 0 {
				assert.Equal(t, "## Workflows\nThe application also runs these workflows for the requests that ask for them. When asked what the application can do, describe them:\n- Restart a service: Restarts a service", body.Messages[1].Content)
			}
			assert.Len(t, req.Messages, 2, "the caller's messages are not edited")
		})
	}
}
