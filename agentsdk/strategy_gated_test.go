package agentsdk_test

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
	agentanthropic "github.com/morebec/proxias/agentsdk/anthropic"
)

func TestGatedStrategy(t *testing.T) {
	tests := []struct {
		name    string
		actions bool
		given   []agentsdk.Event
		// claude are the contents of Claude's answers, in order: the gate's two (the same, as the choice and the noul are
		// asked concurrently), the extraction's, the synthesis'.
		claude []string
		then   []string
		calls  int
		stages []agentsdk.GatedStage
	}{
		{
			name:    "GIVEN the gate is sure of an action THEN its parameters are extracted, it is executed, and the synthesizer replies",
			actions: true,
			given: []agentsdk.Event{
				{ID: "m", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("echo a and b")},
			},
			claude: []string{
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.9}"}]`,
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.9}"}]`,
				`[{"type":"tool_use","id":"toolu_1","name":"echo","input":{"text":"a"}},{"type":"tool_use","id":"toolu_2","name":"echo","input":{"text":"b"}}]`,
				`[{"type":"text","text":"Echoed a and b."}]`,
			},
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"a"}}`,
				`action.request m {"action_id":"echo","params":{"text":"b"}}`,
				`action.result toolu_1 {"action_id":"echo","status":"completed","data":{"text":"a"}}`,
				`action.result toolu_2 {"action_id":"echo","status":"completed","data":{"text":"b"}}`,
				"message  Echoed a and b.",
			},
			calls:  4,
			stages: []agentsdk.GatedStage{agentsdk.GatedClassify, agentsdk.GatedExtract, agentsdk.GatedExecute, agentsdk.GatedSynthesize},
		},
		{
			name:    "GIVEN the gate finds no action is asked for THEN the fallback takes the turn",
			actions: true,
			given: []agentsdk.Event{
				{ID: "m", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("thanks")},
			},
			claude: []string{
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.1}"}]`,
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.1}"}]`,
			},
			then:   []string{"message  fallback"},
			calls:  2,
			stages: []agentsdk.GatedStage{agentsdk.GatedClassify, agentsdk.GatedFallback},
		},
		{
			name:    "GIVEN the gate is not confident of the action THEN the fallback takes the turn",
			actions: true,
			given: []agentsdk.Event{
				{ID: "m", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("do the thing")},
			},
			claude: []string{
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.3,\"noul\":0.9}"}]`,
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.3,\"noul\":0.9}"}]`,
			},
			then:   []string{"message  fallback"},
			calls:  2,
			stages: []agentsdk.GatedStage{agentsdk.GatedClassify, agentsdk.GatedFallback},
		},
		{
			name:    "GIVEN the gate chooses an action the agent does not have THEN the fallback takes the turn",
			actions: true,
			given: []agentsdk.Event{
				{ID: "m", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("delete everything")},
			},
			claude: []string{
				`[{"type":"text","text":"{\"choice\":\"rm_rf\",\"confidence\":0.9,\"noul\":0.9}"}]`,
				`[{"type":"text","text":"{\"choice\":\"rm_rf\",\"confidence\":0.9,\"noul\":0.9}"}]`,
			},
			then:   []string{"message  fallback"},
			calls:  2,
			stages: []agentsdk.GatedStage{agentsdk.GatedClassify, agentsdk.GatedFallback},
		},
		{
			name:    "GIVEN the extraction calls no tool THEN the fallback takes the turn",
			actions: true,
			given: []agentsdk.Event{
				{ID: "m", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("echo")},
			},
			claude: []string{
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.9}"}]`,
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.9}"}]`,
				`[{"type":"text","text":"What should I echo?"}]`,
			},
			then:   []string{"message  fallback"},
			calls:  3,
			stages: []agentsdk.GatedStage{agentsdk.GatedClassify, agentsdk.GatedExtract, agentsdk.GatedFallback},
		},
		{
			name:    "GIVEN the last event is an action result THEN the fallback takes the turn, ungated",
			actions: true,
			given: []agentsdk.Event{
				{ID: "r", Type: agentsdk.EventTypeActionResult, Source: agentsdk.NewSystemSource("actions"), Payload: []byte(`{}`)},
			},
			then:   []string{"message  fallback"},
			stages: []agentsdk.GatedStage{agentsdk.GatedFallback},
		},
		{
			name: "GIVEN the agent has no actions THEN the fallback takes the turn, ungated",
			given: []agentsdk.Event{
				{ID: "m", Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("hi")},
			},
			then:   []string{"message  fallback"},
			stages: []agentsdk.GatedStage{agentsdk.GatedFallback},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int64 // the gate's calls are concurrent
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				content := `[{"type":"text","text":""}]`
				if call := int(calls.Add(1)) - 1; call < len(tt.claude) {
					content = tt.claude[call]
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":` + content + `}`))
			}))
			defer server.Close()
			claude := agentanthropic.NewLLM(anthropicsdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test"), option.WithMaxRetries(0)), "m")
			fallback := agentsdk.StrategyFunc(func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
				return func(yield func(agentsdk.StreamUpdate) bool) {
					yield(agentsdk.StreamUpdate{Events: []agentsdk.Event{agentsdk.NewEvent(agentsdk.EventTypeMessage, agentsdk.NewAgentSource("ada"), []byte("fallback"))}, StopReason: agentsdk.StopReasonEndTurn})
				}
			})
			agent := agentsdk.Agent{ID: "ada"}
			if tt.actions {
				agent.ActionExecutor = agentsdk.SimpleActionExecutor{Actions: map[string]agentsdk.SimpleAction{
					"echo": {ActionDescriptor: agentsdk.ActionDescriptor{ID: "echo", Description: "Echoes the text."}, RunFn: func(_ context.Context, _ agentsdk.Session, params map[string]any) (agentsdk.ActionExecutionResult, error) {
						return agentsdk.ActionExecutionResult{ActionID: "echo", Status: "completed", Data: params}, nil
					}},
				}}
			}
			var stages []agentsdk.GatedStage
			strategy := agentsdk.NewGatedStrategy(agentsdk.LLMSystem1{LLM: claude}, claude, fallback, agentsdk.WithStageTimer(func(stage agentsdk.GatedStage, _ time.Duration) {
				stages = append(stages, stage)
			}))

			var events []string
			for u := range strategy.Invoke(context.Background(), agentsdk.Invocation{Agent: agent, Session: agentsdk.NewSession(tt.given...)}) {
				require.NoError(t, u.Err)
				for _, e := range u.Events {
					events = append(events, string(e.Type)+" "+string(e.CausationID)+" "+e.Text())
				}
			}

			assert.Equal(t, tt.then, events)
			assert.Equal(t, tt.calls, int(calls.Load()))
			assert.Equal(t, tt.stages, stages)
		})
	}
}
