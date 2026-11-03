package agentsdk_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
	agentanthropic "github.com/morebec/proxias/agentsdk/anthropic"
)

func TestLLMSystem1(t *testing.T) {
	tests := []struct {
		name string
		// kind is the question asked: choice, score or noul.
		kind     string
		answer   string
		then     any
		question string
		wantErr  bool
	}{
		{
			name:     "GIVEN a choice THEN the answer is the option, its probabilities and the confidence",
			kind:     "choice",
			answer:   `{"choice":"echo","probabilities":{"echo":0.8,"page":0.2},"confidence":0.8}`,
			then:     agentsdk.System1ChoiceAnswer{Choice: "echo", Probabilities: map[string]float64{"echo": 0.8, "page": 0.2}, Confidence: 0.8},
			question: `{\"instructions\":\"The action asked for.\",\"criteria\":{\"echo\":\"Echoes.\",\"page\":\"Pages.\"}}`,
		},
		{
			name:     "GIVEN a score THEN the answer is the score from 0 to 1, its probabilities and the confidence",
			kind:     "score",
			answer:   `{"score":0.67,"probabilities":{"low":0.1,"high":0.9},"confidence":0.9}`,
			then:     agentsdk.System1ScoreAnswer{Score: 0.67, Probabilities: map[string]float64{"low": 0.1, "high": 0.9}, Confidence: 0.9},
			question: `{\"instructions\":\"How urgent it is.\",\"criteria\":{\"high\":\"Urgent.\",\"low\":\"Not urgent.\"}}`,
		},
		{
			name:     "GIVEN a noul THEN the answer is the probability the statement is true",
			kind:     "noul",
			answer:   `{"noul":0.99}`,
			then:     agentsdk.System1NoulAnswer{Noul: 0.99},
			question: `{\"instructions\":\"The last event asks for an action.\"}`,
		},
		{
			name:     "GIVEN prose around the JSON THEN the JSON is read",
			kind:     "noul",
			answer:   `Here: {"noul":0.1} done`,
			then:     agentsdk.System1NoulAnswer{Noul: 0.1},
			question: `{\"instructions\":\"The last event asks for an action.\"}`,
		},
		{
			name:    "GIVEN no JSON for a choice THEN it is an error",
			kind:    "choice",
			answer:  "echo",
			wantErr: true,
		},
		{
			name:    "GIVEN a score that is not a number THEN it is an error",
			kind:    "score",
			answer:  `{"score":"high"}`,
			wantErr: true,
		},
		{
			name:    "GIVEN no JSON for a noul THEN it is an error",
			kind:    "noul",
			answer:  "yes",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var told map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewDecoder(r.Body).Decode(&told)
				text, _ := json.Marshal(tt.answer)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":` + string(text) + `}]}`))
			}))
			defer server.Close()
			system1 := agentsdk.LLMSystem1{LLM: agentanthropic.NewLLM(anthropicsdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test"), option.WithMaxRetries(0)), "m")}

			var (
				answer any
				err    error
			)
			switch tt.kind {
			case "choice":
				answer, err = system1.AnswerChoice(context.Background(), "john: echo hi", agentsdk.System1Question{Instructions: "The action asked for.", Criteria: map[string]string{"echo": "Echoes.", "page": "Pages."}})
			case "score":
				answer, err = system1.AnswerScore(context.Background(), "john: echo hi", agentsdk.System1Question{Instructions: "How urgent it is.", Criteria: map[string]string{"low": "Not urgent.", "high": "Urgent."}})
			case "noul":
				answer, err = system1.AnswerNoul(context.Background(), "john: echo hi", agentsdk.System1Question{Instructions: "The last event asks for an action."})
			}

			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.then, answer)
			messages, _ := json.Marshal(told["messages"])
			assert.Contains(t, string(messages), "john: echo hi", "the model is told the state")
			assert.Contains(t, string(messages), tt.question, "and the question")
			assert.Nil(t, told["tools"])
		})
	}
}
