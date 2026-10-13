package openai_test

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

var restartedSynthesis = proxias.WorkflowExecutionSynthesis{
	Request:          userSays("restart the api"),
	Intent:           proxias.Intent{WorkflowExecutions: []proxias.WorkflowExecutionIntent{{WorkflowID: "restart_service", Params: map[string]any{"service": "api"}}}},
	ExecutionResults: []proxias.WorkflowExecutionResult{{WorkflowID: "restart_service", Status: "completed", Data: map[string]any{"took": "2s"}}},
}

func TestWorkflowExecutionSynthesizerSynthesize(t *testing.T) {
	tests := []struct {
		name  string
		given string // the text the model writes
		then  assert.ErrorAssertionFunc
	}{
		{
			name:  "canonical",
			given: "The api restarted.",
			then:  assert.NoError,
		},
		{
			name:  "GIVEN the model wrote nothing THEN it fails",
			given: "  ",
			then:  assert.Error,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, _ := mockOpenAI(t, http.StatusOK, answerWith(tt.given))
			synthesizer := openai.NewWorkflowExecutionSynthesizer(client, someModel)

			resp, err := synthesizer.Synthesize(t.Context(), restartedSynthesis)

			tt.then(t, err)
			if err != nil {
				return
			}
			defer func() { _ = resp.Body.Close() }()
			var completion proxias.ChatCompletion
			require.NoError(t, json.NewDecoder(resp.Body).Decode(&completion))
			assert.Equal(t, "chatcmpl-1", completion.ID, "the reply is the model's completion")
			assert.Equal(t, int64(5), completion.Usage.TotalTokens, "its usage included")
			assert.Equal(t, tt.given, completion.Choices[0].Message.Content.Text())
		})
	}
}

func TestWorkflowExecutionSynthesizerSynthesizeStreamsTheReplyWhenAskedFor(t *testing.T) {
	// Given a request that streams
	client, _ := mockOpenAI(t, http.StatusOK, answerWith("The api restarted."))
	synthesizer := openai.NewWorkflowExecutionSynthesizer(client, someModel)
	synthesis := restartedSynthesis
	synthesis.Request.Stream = true

	// When
	resp, err := synthesizer.Synthesize(t.Context(), synthesis)

	// Then
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	assert.Equal(t, "text/event-stream", resp.Header.Get("Content-Type"))
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	assert.Contains(t, string(body), `"content":"The api restarted."`)
}

func TestWorkflowExecutionSynthesizerSynthesizeSendsTheConversationAndTheResults(t *testing.T) {
	// Given
	client, got := mockOpenAI(t, http.StatusOK, answerWith("ok"))
	synthesizer := openai.NewWorkflowExecutionSynthesizer(client, someModel)

	// When
	resp, err := synthesizer.Synthesize(t.Context(), restartedSynthesis)

	// Then
	require.NoError(t, err)
	_ = resp.Body.Close()
	input := sentInput(t, got)
	assert.Contains(t, input, `{"role":"user","content":"restart the api"}`)
	assert.Contains(t, input, `"executions":[{"workflow_id":"restart_service","params":{"service":"api"}}]`)
	assert.Contains(t, input, `"data":{"took":"2s"}`)
}

func TestWorkflowExecutionSynthesizerSynthesizeReturnsTheAPIError(t *testing.T) {
	// Given
	client, _ := mockOpenAI(t, http.StatusTooManyRequests, `{"error":{"type":"rate_limit_error","message":"slow down"}}`)
	synthesizer := openai.NewWorkflowExecutionSynthesizer(client, someModel)

	// When
	_, err := synthesizer.Synthesize(t.Context(), restartedSynthesis)

	// Then
	var apiErr *openaisdk.Error
	require.ErrorAs(t, err, &apiErr)
	assert.ErrorContains(t, err, "openai: synthesize")
}
