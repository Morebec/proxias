package agentsdk_test

import (
	"context"
	"encoding/json"
	"errors"
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

func TestReActStrategy(t *testing.T) {
	tests := []struct {
		name   string
		status int
		claude string
		// actions are the agent's, nil for none.
		actions map[string]agentsdk.SimpleAction
		chunks  []string
		then    []string
		err     string
	}{
		{
			name:   "GIVEN the model answers text THEN it is streamed, and the final output",
			status: http.StatusOK,
			claude: `[{"type":"text","text":"Hello John."}]`,
			chunks: []string{"Hello John."},
			then:   []string{"message agent:ada  Hello John."},
		},
		{
			name:   "GIVEN the model calls a tool THEN its text is a thought, the call is executed, and the turn ends with its result",
			status: http.StatusOK,
			claude: `[{"type":"text","text":"An email."},{"type":"tool_use","id":"toolu_1","name":"send_email","input":{"to":"joe"}}]`,
			actions: map[string]agentsdk.SimpleAction{
				"send_email": {ActionDescriptor: agentsdk.ActionDescriptor{ID: "send_email"}, RunFn: func(_ context.Context, _ agentsdk.Session, params map[string]any) (agentsdk.ActionExecutionResult, error) {
					return agentsdk.ActionExecutionResult{ActionID: "send_email", Status: "completed", Data: params}, nil
				}},
			},
			chunks: []string{"An email."},
			then: []string{
				"cognition.thought agent:ada  An email.",
				`action.request agent:ada  {"action_id":"send_email","params":{"to":"joe"}}`,
				`action.result system:actions toolu_1 {"action_id":"send_email","status":"completed","data":{"to":"joe"}}`,
			},
		},
		{
			name:   "GIVEN the model thinks, then calls a tool THEN its thinking is a thought, kept whole, before the request",
			status: http.StatusOK,
			claude: `[{"type":"thinking","thinking":"An email to joe.","signature":"sig1"},{"type":"tool_use","id":"toolu_1","name":"send_email","input":{"to":"joe"}}]`,
			actions: map[string]agentsdk.SimpleAction{
				"send_email": {ActionDescriptor: agentsdk.ActionDescriptor{ID: "send_email"}, RunFn: func(context.Context, agentsdk.Session, map[string]any) (agentsdk.ActionExecutionResult, error) {
					return agentsdk.ActionExecutionResult{ActionID: "send_email", Status: "completed"}, nil
				}},
			},
			then: []string{
				`cognition.thought agent:ada  {"Text":"","ToolCall":null,"Thinking":"An email to joe.","Signature":"sig1","RedactedThinking":""}`,
				`action.request agent:ada  {"action_id":"send_email","params":{"to":"joe"}}`,
				`action.result system:actions toolu_1 {"action_id":"send_email","status":"completed"}`,
			},
		},
		{
			name:   "GIVEN the execution of a call fails THEN the error ends the turn, after the request",
			status: http.StatusOK,
			claude: `[{"type":"tool_use","id":"toolu_1","name":"send_email","input":{"to":"joe"}}]`,
			actions: map[string]agentsdk.SimpleAction{
				"send_email": {ActionDescriptor: agentsdk.ActionDescriptor{ID: "send_email"}, RunFn: func(context.Context, agentsdk.Session, map[string]any) (agentsdk.ActionExecutionResult, error) {
					return agentsdk.ActionExecutionResult{}, errors.New("smtp down")
				}},
			},
			then: []string{`action.request agent:ada  {"action_id":"send_email","params":{"to":"joe"}}`},
			err:  "executing actions: send_email: smtp down",
		},
		{
			name:   "GIVEN the model calls a tool of an agent without actions THEN it is an error",
			status: http.StatusOK,
			claude: `[{"type":"tool_use","id":"toolu_1","name":"send_email","input":{"to":"joe"}}]`,
			then:   []string{`action.request agent:ada  {"action_id":"send_email","params":{"to":"joe"}}`},
			err:    "the model called tools, but the agent has no actions",
		},
		{
			name:   "GIVEN the model answers nothing THEN the final output is empty: silence",
			status: http.StatusOK,
			claude: `[]`,
			then:   []string{"message agent:ada  "},
		},
		{
			name:   "GIVEN the model fails THEN it is an error",
			status: http.StatusInternalServerError,
			claude: `[]`,
			err:    "500 Internal Server Error",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":` + tt.claude + `}`))
			}))
			defer server.Close()
			llm := agentanthropic.NewLLM(anthropicsdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test"), option.WithMaxRetries(0)), "m")
			agent := agentsdk.Agent{ID: "ada"}
			if tt.actions != nil {
				agent.ActionExecutor = agentsdk.SimpleActionExecutor{Actions: tt.actions}
			}
			inv := agentsdk.Invocation{Agent: agent, Session: agentsdk.NewSession(agentsdk.NewMessageEvent(agentsdk.NewUserSource("john"), "hi"))}

			var (
				chunks []string
				events []string
				err    error
			)
			for u := range agentsdk.NewReActStrategy(llm).Invoke(context.Background(), inv) {
				if u.Err != nil {
					err = u.Err
					continue
				}
				if u.Chunk != nil {
					chunks = append(chunks, string(u.Chunk.Data))
				}
				for _, e := range u.Events {
					assert.NotEmpty(t, e.ID, "the events are whole")
					assert.False(t, e.Date.IsZero(), "the events are dated")
					events = append(events, string(e.Type)+" "+e.Source.String()+" "+string(e.CausationID)+" "+e.Text())
				}
			}

			if tt.err != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, tt.chunks, chunks)
			assert.Equal(t, tt.then, events)
		})
	}
}

func TestReActStrategyRequest(t *testing.T) {
	tests := []struct {
		name     string
		given    []agentsdk.Event
		system   []string
		messages string
	}{
		{
			name: "GIVEN leading system messages THEN they are instructions",
			given: []agentsdk.Event{
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewSystemSource("app"), Payload: []byte("Be brief.")},
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("hi")},
			},
			system:   []string{"Be brief."},
			messages: `[{"role":"user","content":[{"type":"text","text":"hi"}]}]`,
		},
		{
			name:   "GIVEN the agent's action and its result THEN they are a tool call and its result",
			system: []string{},
			given: []agentsdk.Event{
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("email joe")},
				{ID: "toolu_1", Type: agentsdk.EventTypeActionRequest, Source: agentsdk.NewAgentSource("ada"), Payload: []byte(`{"action_id":"send_email","params":{"to":"joe"}}`)},
				{Type: agentsdk.EventTypeActionResult, Source: agentsdk.NewSystemSource("actions"), CausationID: "toolu_1", Payload: []byte(`{"status":"completed"}`)},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"email joe"}]},
				{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"send_email","input":{"to":"joe"}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"{\"status\":\"completed\"}"}],"is_error":false}]}
			]`,
		},
		{
			name:   "GIVEN the agent's thinking, its action and the result THEN the thinking is given back signed before the tool call",
			system: []string{},
			given: []agentsdk.Event{
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewUserSource("john"), Payload: []byte("email joe")},
				{ID: "t", Type: agentsdk.EventTypeThought, Source: agentsdk.NewAgentSource("ada"), Payload: []byte(`{"Thinking":"An email to joe.","Signature":"sig1"}`)},
				{ID: "toolu_1", Type: agentsdk.EventTypeActionRequest, Source: agentsdk.NewAgentSource("ada"), Payload: []byte(`{"action_id":"send_email","params":{"to":"joe"}}`)},
				{Type: agentsdk.EventTypeActionResult, Source: agentsdk.NewSystemSource("actions"), CausationID: "toolu_1", Payload: []byte(`{"status":"completed"}`)},
			},
			messages: `[
				{"role":"user","content":[{"type":"text","text":"email joe"}]},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"An email to joe.","signature":"sig1"},
					{"type":"tool_use","id":"toolu_1","name":"send_email","input":{"to":"joe"}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"{\"status\":\"completed\"}"}],"is_error":false}]}
			]`,
		},
		{
			name:   "GIVEN events of other sources THEN they are tagged with their type and source",
			system: []string{},
			given: []agentsdk.Event{
				{Type: "probe.disk", Source: agentsdk.NewSystemSource("db-1"), Payload: []byte("93%")},
				{Type: agentsdk.EventTypeMessage, Source: agentsdk.NewAgentSource("rita"), Payload: []byte("hi")},
			},
			messages: `[{"role":"user","content":[{"type":"text","text":"[probe.disk from system:db-1] 93%"},{"type":"text","text":"[message from agent:rita] hi"}]}]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var told map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewDecoder(r.Body).Decode(&told)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"ok"}]}`))
			}))
			defer server.Close()
			llm := agentanthropic.NewLLM(anthropicsdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test"), option.WithMaxRetries(0)), "m")
			agent := agentsdk.Agent{ID: "ada", AgentPersona: agentsdk.AgentPersona{Name: "Ada"}, ActionExecutor: agentsdk.SimpleActionExecutor{Actions: map[string]agentsdk.SimpleAction{
				"send_email": {ActionDescriptor: agentsdk.ActionDescriptor{ID: "send_email", Description: "Sends an email."}},
			}}}

			for u := range agentsdk.NewReActStrategy(llm).Invoke(context.Background(), agentsdk.Invocation{Agent: agent, Session: agentsdk.NewSession(tt.given...)}) {
				require.NoError(t, u.Err)
			}

			var system []struct {
				Text string `json:"text"`
			}
			encoded, _ := json.Marshal(told["system"])
			require.NoError(t, json.Unmarshal(encoded, &system))
			require.NotEmpty(t, system)
			assert.Contains(t, system[0].Text, agentsdk.NewAgentPrompt(agent), "the model is told who the agent is")
			instructions := make([]string, 0, len(system)-1)
			for _, block := range system[1:] {
				instructions = append(instructions, block.Text)
			}
			assert.Equal(t, tt.system, instructions)
			messages, _ := json.Marshal(told["messages"])
			assert.JSONEq(t, tt.messages, string(messages))
			tools, _ := json.Marshal(told["tools"])
			assert.Contains(t, string(tools), `"name":"send_email"`, "the agent's actions are tools")
		})
	}
}
