package agentprotocolhttp_test

import (
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias/agentsdk"
	"github.com/morebec/proxias/agentsdk/transports/agentprotocolhttp"
)

func TestHandlerSteps(t *testing.T) {
	tests := []struct {
		name  string
		turns [][]agentsdk.Event
		task  string
		steps []string
		then  []agentprotocolhttp.Step
		told  []string
	}{
		{
			name: "GIVEN a turn that replies THEN the step is the last, its output the reply",
			turns: [][]agentsdk.Event{
				{agentsdk.NewEvent(agentsdk.EventTypeMessage, agentsdk.NewAgentSource("ada"), []byte("Hi."))},
			},
			task:  `{"input":"hello"}`,
			steps: []string{``},
			then:  []agentprotocolhttp.Step{{Name: "turn 1", Status: "completed", Output: "Hi.", IsLast: true, Artifacts: []any{}}},
			told:  []string{"user:user hello"},
		},
		{
			name: "GIVEN a turn that ends on the results of its actions THEN the step is not the last, and the next step takes the next turn",
			turns: [][]agentsdk.Event{
				{
					agentsdk.NewEvent(agentsdk.EventTypeThought, agentsdk.NewAgentSource("ada"), []byte("Echoing.")),
					{ID: "q", Type: agentsdk.EventTypeActionRequest, Source: agentsdk.NewAgentSource("ada"), Payload: []byte(`{"action_id":"echo"}`), Date: time.Now()},
					{ID: "r", CausationID: "q", Type: agentsdk.EventTypeActionResult, Source: agentsdk.NewSystemSource("actions"), Payload: []byte(`{"action_id":"echo","status":"completed"}`), Date: time.Now()},
				},
				{agentsdk.NewEvent(agentsdk.EventTypeMessage, agentsdk.NewAgentSource("ada"), []byte("Echoed."))},
			},
			task:  `{"input":"echo"}`,
			steps: []string{``, `{"input":"thanks"}`},
			then: []agentprotocolhttp.Step{
				{Name: "turn 1", Status: "completed", Output: "Echoing.", AdditionalOutput: map[string]any{"actions": float64(1)}, Artifacts: []any{}},
				{Name: "turn 2", Status: "completed", Input: "thanks", Output: "Echoed.", IsLast: true, Artifacts: []any{}},
			},
			told: []string{
				"user:user echo",
				"user:user echo", "agent:ada Echoing.", `agent:ada {"action_id":"echo"}`, `system:actions {"action_id":"echo","status":"completed"}`, "user:user thanks",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				told []string
				turn int
			)
			strategy := agentsdk.StrategyFunc(func(_ context.Context, inv agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
				for _, e := range inv.Session.Events() {
					told = append(told, e.Source.String()+" "+e.Text())
				}
				return func(yield func(agentsdk.StreamUpdate) bool) {
					turn++
					var stop agentsdk.StopReason
					if events := tt.turns[turn-1]; events[len(events)-1].Type == agentsdk.EventTypeMessage {
						stop = agentsdk.StopReasonEndTurn
					}
					yield(agentsdk.StreamUpdate{Events: tt.turns[turn-1], StopReason: stop})
				}
			})
			handler := agentprotocolhttp.NewHandler(agentsdk.Agent{ID: "ada"}, strategy)

			w := httptest.NewRecorder()
			handler.ServeHTTP(w, httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/ap/v1/agent/tasks", strings.NewReader(tt.task)))
			require.Equal(t, http.StatusOK, w.Code)
			var task agentprotocolhttp.Task
			require.NoError(t, json.Unmarshal(w.Body.Bytes(), &task))
			var steps []agentprotocolhttp.Step
			for _, body := range tt.steps {
				w := httptest.NewRecorder()
				handler.ServeHTTP(w, httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/ap/v1/agent/tasks/"+task.TaskID+"/steps", strings.NewReader(body)))
				require.Equal(t, http.StatusOK, w.Code)
				var step agentprotocolhttp.Step
				require.NoError(t, json.Unmarshal(w.Body.Bytes(), &step))
				assert.Equal(t, task.TaskID, step.TaskID)
				assert.NotEmpty(t, step.StepID)
				step.TaskID, step.StepID = "", ""
				steps = append(steps, step)
			}

			assert.Equal(t, tt.then, steps)
			assert.Equal(t, tt.told, told)
		})
	}
}

func TestHandlerReads(t *testing.T) {
	tests := []struct {
		name string
		when func(taskID, stepID string) string
		code int
		then string
	}{
		{
			name: "WHEN the tasks THEN the task is listed",
			when: func(string, string) string { return "/ap/v1/agent/tasks" },
			code: http.StatusOK,
			then: `{"tasks":[{"task_id":"TASK","input":"hello","artifacts":[]}],"pagination":{"total_items":1,"total_pages":1,"current_page":1,"page_size":1}}`,
		},
		{
			name: "WHEN the task THEN it is the task",
			when: func(taskID, _ string) string { return "/ap/v1/agent/tasks/" + taskID },
			code: http.StatusOK,
			then: `{"task_id":"TASK","input":"hello","artifacts":[]}`,
		},
		{
			name: "WHEN the steps THEN the step is listed",
			when: func(taskID, _ string) string { return "/ap/v1/agent/tasks/" + taskID + "/steps" },
			code: http.StatusOK,
			then: `{"steps":[{"task_id":"TASK","step_id":"STEP","name":"turn 1","status":"completed","output":"Hi.","artifacts":[],"is_last":true}],"pagination":{"total_items":1,"total_pages":1,"current_page":1,"page_size":1}}`,
		},
		{
			name: "WHEN the step THEN it is the step",
			when: func(taskID, stepID string) string { return "/ap/v1/agent/tasks/" + taskID + "/steps/" + stepID },
			code: http.StatusOK,
			then: `{"task_id":"TASK","step_id":"STEP","name":"turn 1","status":"completed","output":"Hi.","artifacts":[],"is_last":true}`,
		},
		{
			name: "WHEN the artifacts THEN there are none",
			when: func(taskID, _ string) string { return "/ap/v1/agent/tasks/" + taskID + "/artifacts" },
			code: http.StatusOK,
			then: `{"artifacts":[],"pagination":{"total_items":0,"total_pages":1,"current_page":1,"page_size":1}}`,
		},
		{
			name: "WHEN an unknown task THEN it is not found",
			when: func(string, string) string { return "/ap/v1/agent/tasks/nope" },
			code: http.StatusNotFound,
			then: `{"message":"no task nope"}`,
		},
		{
			name: "WHEN an unknown step THEN it is not found",
			when: func(taskID, _ string) string { return "/ap/v1/agent/tasks/" + taskID + "/steps/nope" },
			code: http.StatusNotFound,
			then: `{"message":"no step nope"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := agentsdk.StrategyFunc(func(context.Context, agentsdk.Invocation) iter.Seq[agentsdk.StreamUpdate] {
				return func(yield func(agentsdk.StreamUpdate) bool) {
					yield(agentsdk.StreamUpdate{Events: []agentsdk.Event{agentsdk.NewEvent(agentsdk.EventTypeMessage, agentsdk.NewAgentSource("ada"), []byte("Hi."))}, StopReason: agentsdk.StopReasonEndTurn})
				}
			})
			handler := agentprotocolhttp.NewHandler(agentsdk.Agent{ID: "ada"}, strategy)
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/ap/v1/agent/tasks", strings.NewReader(`{"input":"hello"}`)))
			var task agentprotocolhttp.Task
			require.NoError(t, json.Unmarshal(w.Body.Bytes(), &task))
			w = httptest.NewRecorder()
			handler.ServeHTTP(w, httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/ap/v1/agent/tasks/"+task.TaskID+"/steps", nil))
			var step agentprotocolhttp.Step
			require.NoError(t, json.Unmarshal(w.Body.Bytes(), &step))

			w = httptest.NewRecorder()
			handler.ServeHTTP(w, httptest.NewRequestWithContext(t.Context(), http.MethodGet, tt.when(task.TaskID, step.StepID), nil))

			assert.Equal(t, tt.code, w.Code)
			body := strings.ReplaceAll(strings.ReplaceAll(w.Body.String(), task.TaskID, "TASK"), step.StepID, "STEP")
			assert.JSONEq(t, tt.then, body)
		})
	}
}
