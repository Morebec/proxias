package agentsdk_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
	"github.com/morebec/proxias/agentsdk/adapters"
	agentanthropic "github.com/morebec/proxias/agentsdk/adapters/anthropic"
)

func TestGatedStrategy(t *testing.T) {
	tests := []struct {
		name string
		// single classifies with an LLMActionClassifier, otherwise a
		// CompositeActionClassifier: system1 selects with a
		// System1ActionSelector, otherwise an LLMActionSelector.
		single, system1 bool
		// selectionThreshold is the System1 selector's, nil for the default.
		selectionThreshold *float64
		actions            bool
		// The prompts of the single-call classifier, the selector, the
		// extractor and the synthesis.
		classificationPrompt, selectionPrompt, extractionPrompt, synthesisPrompt string
		given                                                                    []agentsdk.Event
		// claude are the contents of Claude's answers, in order. The System1
		// selector's choice and noul are asked
		// concurrently: their answers are the same.
		claude []string
		// stopReasons are Claude's stop_reason of each call, end_turn when
		// missing; stop the turn's, checked when set.
		stopReasons []string
		stop        agentsdk.StopReason
		then        []string
		// prompts are the first system prompt of each call, in any order.
		prompts []string
		// memories are what the agent's memory recalls; remembered, when set, is in
		// the last system prompt of every call.
		memories   []agentsdk.Event
		remembered string
	}{
		{
			name:    "GIVEN the synthesizer declines to reply THEN the turn ends with the content filter, after the executions",
			single:  true,
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo a"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"a\"}}]}"}]`,
				`[]`,
			},
			stopReasons: []string{"end_turn", "refusal"},
			stop:        agentsdk.StopReasonContentFilter,
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"a"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"a"}}`,
				"message  ",
			},
			prompts: []string{
				agentsdk.DefaultLLMClassificationPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN the synthesizer is cut at the limit of tokens THEN the turn ends with the length",
			single:  true,
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo a"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"a\"}}]}"}]`,
				`[{"type":"text","text":"Echoed"}]`,
			},
			stopReasons: []string{"end_turn", "max_tokens"},
			stop:        agentsdk.StopReasonLength,
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"a"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"a"}}`,
				"message  Echoed",
			},
			prompts: []string{
				agentsdk.DefaultLLMClassificationPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN memories THEN the single-call classifier and the synthesis are told them, last",
			single:  true,
			actions: true,
			memories: []agentsdk.Event{
				agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "joe is joe@boss.com"),
			},
			remembered: "joe is joe@boss.com",
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo joe's email"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"joe@boss.com\"}}]}"}]`,
				`[{"type":"text","text":"Echoed."}]`,
			},
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"joe@boss.com"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"joe@boss.com"}}`,
				"message  Echoed.",
			},
			prompts: []string{
				agentsdk.DefaultLLMClassificationPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN memories THEN the selector, the extractor and the synthesis are told them, last",
			actions: true,
			memories: []agentsdk.Event{
				agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "joe is joe@boss.com"),
			},
			remembered: "joe is joe@boss.com",
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo joe's email"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"actions\":[\"echo\"]}"}]`,
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"joe@boss.com\"}}]}"}]`,
				`[{"type":"text","text":"Echoed."}]`,
			},
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"joe@boss.com"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"joe@boss.com"}}`,
				"message  Echoed.",
			},
			prompts: []string{
				agentsdk.DefaultLLMSelectionPrompt,
				agentsdk.DefaultLLMExtractionPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN the single-call classifier finds executions THEN they are executed, and the synthesizer replies",
			single:  true,
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo a and b"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"a\"}},{\"action_id\":\"echo\",\"params\":{\"text\":\"b\"}}]}"}]`,
				`[{"type":"text","text":"Echoed a and b."}]`,
			},
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"a"}}`,
				`action.request m {"action_id":"echo","params":{"text":"b"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"a"}}`,
				`action.result request 2 {"action_id":"echo","status":"completed","data":{"text":"b"}}`,
				"message  Echoed a and b.",
			},
			prompts: []string{
				agentsdk.DefaultLLMClassificationPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN instructions THEN they follow the classifier's own prompt and inventory",
			single:  true,
			actions: true,
			given: []agentsdk.Event{
				{
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewSystemSource("app"),
					Payload: []byte("Be brief."),
				},
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("thanks"),
				},
			},
			claude:  []string{`[{"type":"text","text":"{\"executions\":[]}"}]`},
			then:    []string{"message  reasoning"},
			prompts: []string{agentsdk.DefaultLLMClassificationPrompt},
		},
		{
			name:                 "GIVEN a prompt of the single-call classifier THEN it is the system prompt of its call",
			single:               true,
			actions:              true,
			classificationPrompt: "Classify.",
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("thanks"),
				},
			},
			claude:  []string{`[{"type":"text","text":"{\"executions\":[]}"}]`},
			then:    []string{"message  reasoning"},
			prompts: []string{"Classify."},
		},
		{
			name:    "GIVEN the single-call classifier answers words with braces after its JSON THEN only its first JSON value is read",
			single:  true,
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo a"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"a\"}}]}\nThe {text} is a."}]`,
				`[{"type":"text","text":"Echoed a."}]`,
			},
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"a"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"a"}}`,
				"message  Echoed a.",
			},
			prompts: []string{
				agentsdk.DefaultLLMClassificationPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN the single-call classifier finds an action the agent does not have THEN it is executed, fails, and the synthesizer replies",
			single:  true,
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("delete everything"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"rm_rf\"}]}"}]`,
				`[{"type":"text","text":"I cannot."}]`,
			},
			then: []string{
				`action.request m {"action_id":"rm_rf"}`,
				`action.result request 1 {"action_id":"rm_rf","status":"failed","data":{"error":"unknown action"}}`,
				"message  I cannot.",
			},
			prompts: []string{
				agentsdk.DefaultLLMClassificationPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN the selector selects an action, and the extractor its executions THEN they are executed, and the synthesizer replies",
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo a and b"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"actions\":[\"echo\"]}"}]`,
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"a\"}},{\"action_id\":\"echo\",\"params\":{\"text\":\"b\"}}]}"}]`,
				`[{"type":"text","text":"Echoed a and b."}]`,
			},
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"a"}}`,
				`action.request m {"action_id":"echo","params":{"text":"b"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"a"}}`,
				`action.result request 2 {"action_id":"echo","status":"completed","data":{"text":"b"}}`,
				"message  Echoed a and b.",
			},
			prompts: []string{
				agentsdk.DefaultLLMSelectionPrompt,
				agentsdk.DefaultLLMExtractionPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:             "GIVEN prompts of the selector, the extractor and the synthesis THEN they are the system prompts of their calls",
			actions:          true,
			selectionPrompt:  "Select.",
			extractionPrompt: "Extract.",
			synthesisPrompt:  "Synthesize.",
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo a"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"actions\":[\"echo\"]}"}]`,
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"a\"}}]}"}]`,
				`[{"type":"text","text":"Echoed a."}]`,
			},
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"a"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"a"}}`,
				"message  Echoed a.",
			},
			prompts: []string{"Select.", "Extract.", "Synthesize."},
		},
		{
			name:    "GIVEN the extractor answers words with braces after its JSON THEN only its first JSON value is read",
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo a"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"actions\":[\"echo\"]}"}]`,
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"a\"}}]}\nThe {text} is a."}]`,
				`[{"type":"text","text":"Echoed a."}]`,
			},
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"a"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"a"}}`,
				"message  Echoed a.",
			},
			prompts: []string{
				agentsdk.DefaultLLMSelectionPrompt,
				agentsdk.DefaultLLMExtractionPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN the selector selects nothing THEN nothing is extracted, and the reasoning takes the turn",
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("thanks"),
				},
			},
			claude:  []string{`[{"type":"text","text":"{\"actions\":[]}"}]`},
			then:    []string{"message  reasoning"},
			prompts: []string{agentsdk.DefaultLLMSelectionPrompt},
		},
		{
			name:    "GIVEN the selector selects an action the agent does not have THEN its parameters are extracted, it is executed, fails, and the synthesizer replies",
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("delete everything"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"actions\":[\"rm_rf\"]}"}]`,
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"rm_rf\"}]}"}]`,
				`[{"type":"text","text":"I cannot."}]`,
			},
			then: []string{
				`action.request m {"action_id":"rm_rf"}`,
				`action.result request 1 {"action_id":"rm_rf","status":"failed","data":{"error":"unknown action"}}`,
				"message  I cannot.",
			},
			prompts: []string{
				agentsdk.DefaultLLMSelectionPrompt,
				agentsdk.DefaultLLMExtractionPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN the extractor gives an execution of an action that was not selected THEN it is executed, fails, and the synthesizer replies",
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo a"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"actions\":[\"echo\"]}"}]`,
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"rm_rf\"}]}"}]`,
				`[{"type":"text","text":"I cannot."}]`,
			},
			then: []string{
				`action.request m {"action_id":"rm_rf"}`,
				`action.result request 1 {"action_id":"rm_rf","status":"failed","data":{"error":"unknown action"}}`,
				"message  I cannot.",
			},
			prompts: []string{
				agentsdk.DefaultLLMSelectionPrompt,
				agentsdk.DefaultLLMExtractionPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN a System1 selector sure of an action THEN the extractor gives its executions, they are executed, and the synthesizer replies",
			system1: true,
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("echo a"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.9}"}]`,
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.9}"}]`,
				`[{"type":"text","text":"{\"executions\":[{\"action_id\":\"echo\",\"params\":{\"text\":\"a\"}}]}"}]`,
				`[{"type":"text","text":"Echoed a."}]`,
			},
			then: []string{
				`action.request m {"action_id":"echo","params":{"text":"a"}}`,
				`action.result request 1 {"action_id":"echo","status":"completed","data":{"text":"a"}}`,
				"message  Echoed a.",
			},
			prompts: []string{
				adapters.DefaultChoicePrompt,
				adapters.DefaultNoulPrompt,
				agentsdk.DefaultLLMExtractionPrompt,
				agentsdk.DefaultSynthesisPrompt,
			},
		},
		{
			name:    "GIVEN a System1 selector that finds no action is asked for THEN the reasoning takes the turn",
			system1: true,
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("thanks"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.1}"}]`,
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.9,\"noul\":0.1}"}]`,
			},
			then: []string{"message  reasoning"},
			prompts: []string{
				adapters.DefaultChoicePrompt,
				adapters.DefaultNoulPrompt,
			},
		},
		{
			name:    "GIVEN a System1 selector not confident of the action THEN the reasoning takes the turn",
			system1: true,
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("do the thing"),
				},
			},
			claude: []string{
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.3,\"noul\":0.9}"}]`,
				`[{"type":"text","text":"{\"choice\":\"echo\",\"confidence\":0.3,\"noul\":0.9}"}]`,
			},
			then: []string{"message  reasoning"},
			prompts: []string{
				adapters.DefaultChoicePrompt,
				adapters.DefaultNoulPrompt,
			},
		},
		{
			name:    "GIVEN the last event is an action result THEN the reasoning takes the turn, unclassified",
			actions: true,
			given: []agentsdk.Event{
				{
					ID:      "r",
					Type:    agentsdk.EventTypeActionResult,
					Source:  agentsdk.NewSystemSource("actions"),
					Payload: []byte(`{}`),
				},
			},
			then: []string{"message  reasoning"},
		},
		{
			name: "GIVEN the agent has no actions THEN the reasoning takes the turn, unclassified",
			given: []agentsdk.Event{
				{
					ID:      "m",
					Type:    agentsdk.EventTypeMessage,
					Source:  agentsdk.NewUserSource("john"),
					Payload: []byte("hi"),
				},
			},
			then: []string{"message  reasoning"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				mutex   sync.Mutex // the System1 selector's calls are concurrent
				prompts []string
				lasts   []string
			)
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var request struct {
						System []struct {
							Text string `json:"text"`
						} `json:"system"`
					}
					body, _ := io.ReadAll(r.Body)
					_ = json.Unmarshal(body, &request)
					mutex.Lock()
					call := len(prompts)
					prompts = append(prompts, request.System[0].Text)
					lasts = append(lasts, request.System[len(request.System)-1].Text)
					mutex.Unlock()
					content := `[{"type":"text","text":""}]`
					if call < len(tt.claude) {
						content = tt.claude[call]
					}
					stopReason := "end_turn"
					if call < len(tt.stopReasons) {
						stopReason = tt.stopReasons[call]
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(
						[]byte(
							`{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"` + stopReason + `","usage":{},"content":` + content + `}`,
						),
					)
				}),
			)
			defer server.Close()
			claude := agentanthropic.NewLLM(
				anthropicsdk.NewClient(
					option.WithBaseURL(server.URL),
					option.WithAPIKey("test"),
					option.WithMaxRetries(0),
				),
				"m",
			)
			reasoning := agentsdk.StrategyFunc(
				func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
					return func(yield func(agentsdk.StreamUpdate) bool) {
						yield(
							agentsdk.StreamUpdate{
								Events: []agentsdk.Event{
									agentsdk.NewEvent(
										agentsdk.EventTypeMessage,
										agentsdk.NewAgentSource("ada"),
										[]byte("reasoning"),
									),
								},
								StopReason: agentsdk.StopReasonEndTurn,
							},
						)
					}
				},
			)
			agent := agentsdk.Agent{ID: "ada"}
			if tt.memories != nil {
				agent.Memory = memory{
					recall: func(agentsdk.Invocation) ([]agentsdk.Event, error) { return tt.memories, nil },
				}
			}
			if tt.actions {
				agent.ActionExecutor = agentsdk.SimpleActionExecutor{
					Actions: map[string]agentsdk.SimpleAction{
						"echo": {
							ActionDescriptor: agentsdk.ActionDescriptor{
								ID:          "echo",
								Description: "Echoes the text.",
							},
							RunFn: func(_ context.Context, _ agentsdk.Session, params map[string]any) (agentsdk.ActionExecutionResult, error) {
								return agentsdk.ActionExecutionResult{
									ActionID: "echo",
									Status:   "completed",
									Data:     params,
								}, nil
							},
						},
					},
				}
			}
			var selector agentsdk.ActionSelector = agentsdk.LLMActionSelector{LLM: claude, Prompt: tt.selectionPrompt}
			if tt.system1 {
				selector = agentsdk.System1ActionSelector{
					System1:            adapters.LLMSystem1{LLM: claude},
					SelectionThreshold: tt.selectionThreshold,
				}
			}
			var classifier agentsdk.ActionClassifier = agentsdk.NewCompositeActionClassifier(selector, agentsdk.LLMActionParameterExtractor{LLM: claude, Prompt: tt.extractionPrompt})
			if tt.single {
				classifier = agentsdk.LLMActionClassifier{
					LLM:    claude,
					Prompt: tt.classificationPrompt,
				}
			}
			strategy := agentsdk.NewGatedStrategy(classifier, claude, reasoning)
			strategy.SynthesisPrompt = tt.synthesisPrompt

			var (
				events []string
				stop   agentsdk.StopReason
			)
			requests := map[agentsdk.EventID]string{} // the requests' IDs are random: a result names its request by its rank
			for u := range strategy.Invoke(context.Background(), agentsdk.Invocation{Agent: agent, Session: agentsdk.NewSession(tt.given...), Memories: tt.memories}) {
				require.NoError(t, u.Err)
				if u.StopReason != "" {
					stop = u.StopReason
				}
				for _, e := range u.Events {
					causation := string(e.CausationID)
					if rank, ok := requests[e.CausationID]; ok {
						causation = rank
					}
					if e.Type == agentsdk.EventTypeActionRequest {
						requests[e.ID] = fmt.Sprintf("request %d", len(requests)+1)
					}
					events = append(events, string(e.Type)+" "+causation+" "+e.Text())
				}
			}

			assert.Equal(t, tt.then, events)
			if tt.stop != "" {
				assert.Equal(t, tt.stop, stop)
			}
			assert.ElementsMatch(t, tt.prompts, prompts)
			for _, last := range lasts {
				if tt.remembered != "" {
					assert.Contains(t, last, tt.remembered)
				}
			}
		})
	}
}
