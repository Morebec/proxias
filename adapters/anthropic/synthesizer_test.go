package anthropic_test

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/anthropic"
)

var restartedSynthesis = proxias.WorkflowExecutionSynthesis{
	Request: userSays("restart api"),
	Intent:  proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{restartIntent}},
	ExecutionResults: []proxias.WorkflowExecutionResult{
		{WorkflowID: "restart_service", Status: "completed", Data: map[string]any{"restarted": true}},
	},
}

func TestWorkflowExecutionSynthesizerSynthesize(t *testing.T) {
	type expect struct {
		body string // the chat completion, its created time zeroed
		err  assert.ErrorAssertionFunc
	}
	tests := []struct {
		name  string
		given string // the text the model writes
		then  expect
	}{
		{
			name:  "canonical",
			given: "The api restarted.",
			then: expect{
				body: `{"id":"msg_1","object":"chat.completion","created":0,"model":"claude-x","choices":[{"index":0,"message":{"role":"assistant","content":"The api restarted."},"finish_reason":"stop"}],
					"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"prompt_tokens_details":{"cached_tokens":0}}}`,
				err: assert.NoError,
			},
		},
		{
			name:  "GIVEN the model writes nothing THEN it fails",
			given: "  ",
			then:  expect{err: assert.Error},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, _ := mockAnthropic(t, 200, answerWith(tt.given))
			synthesizer := anthropic.NewWorkflowExecutionSynthesizer(client, someModel)

			got, err := synthesizer.Synthesize(t.Context(), restartedSynthesis)

			tt.then.err(t, err)
			if err != nil {
				assert.Nil(t, got.Body)
				return
			}
			defer func() { _ = got.Body.Close() }()
			assert.Equal(t, http.StatusOK, got.Status)
			assert.Equal(t, "application/json", got.Header.Get("Content-Type"))
			var completion proxias.ChatCompletion
			require.NoError(t, json.NewDecoder(got.Body).Decode(&completion))
			assert.NotZero(t, completion.Created)
			completion.Created = 0
			raw, err := json.Marshal(completion)
			require.NoError(t, err)
			assert.JSONEq(t, tt.then.body, string(raw))
		})
	}
}

func TestWorkflowExecutionSynthesizerSynthesizeStreamsTheReplyWhenAskedFor(t *testing.T) {
	// Given a request that asks for the usage in the stream
	client, _ := mockAnthropic(t, 200, answerWith("The api restarted."))
	synthesizer := anthropic.NewWorkflowExecutionSynthesizer(client, someModel)
	synthesis := restartedSynthesis
	synthesis.Request.Stream = true
	synthesis.Request.StreamOptions = &proxias.ChatStreamOptions{IncludeUsage: true}

	// When
	got, err := synthesizer.Synthesize(t.Context(), synthesis)

	// Then the reply comes as a stream: the message, finish, usage, end
	require.NoError(t, err)
	defer func() { _ = got.Body.Close() }()
	assert.Equal(t, "text/event-stream", got.Header.Get("Content-Type"))
	raw, err := io.ReadAll(got.Body)
	require.NoError(t, err)
	events := strings.Split(strings.TrimSpace(string(raw)), "\n\n")
	require.Len(t, events, 4)
	assert.Contains(t, events[0], `"delta":{"role":"assistant","content":"The api restarted."}`)
	assert.Contains(t, events[1], `"finish_reason":"stop"`)
	assert.Contains(t, events[2], `"usage":`)
	assert.Equal(t, "data: [DONE]", events[3])
}

func TestWorkflowExecutionSynthesizerSynthesizeSendsTheConversationAndTheResults(t *testing.T) {
	// Given
	client, sent := mockAnthropic(t, 200, answerWith("ok"))
	synthesizer := anthropic.NewWorkflowExecutionSynthesizer(client, someModel)

	// When
	_, err := synthesizer.Synthesize(t.Context(), restartedSynthesis)

	// Then
	require.NoError(t, err)
	var call struct {
		System []struct {
			CacheControl map[string]string `json:"cache_control"`
		} `json:"system"`
		Messages []struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal([]byte(*sent), &call))
	assert.Equal(t, "ephemeral", call.System[0].CacheControl["type"], "the instructions never vary: they can be cached")
	user := call.Messages[0].Content[0].Text
	assert.Contains(t, user, `{"role":"user","content":"restart api"}`)
	assert.Contains(t, user, `"executions":[{"workflow_id":"restart_service","params":{"service":"api"}}]`)
	assert.Contains(t, user, `"results":[{"workflow_id":"restart_service","status":"completed","data":{"restarted":true}}]`)
}

func TestWorkflowExecutionSynthesizerSynthesizeReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockAnthropic(t, 500, `{"type":"error","error":{"type":"api_error","message":"down"}}`)
	synthesizer := anthropic.NewWorkflowExecutionSynthesizer(client, someModel)

	// When
	_, err := synthesizer.Synthesize(t.Context(), restartedSynthesis)

	// Then
	var apiErr *anthropicsdk.Error
	require.ErrorAs(t, err, &apiErr)
	assert.ErrorContains(t, err, "anthropic: synthesize")
}
