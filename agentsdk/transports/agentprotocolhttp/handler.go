// Package agentprotocolhttp serves an agent through the Agent Protocol (agentprotocol.ai, v1): a task is a session,
// a step is one turn of the agent on it.
//
//	POST /ap/v1/agent/tasks                                   creates a task: its input is the session's first message
//	GET  /ap/v1/agent/tasks                                   the tasks
//	GET  /ap/v1/agent/tasks/{task_id}                         a task
//	POST /ap/v1/agent/tasks/{task_id}/steps                   executes a step: its input, if any, is appended as a
//	                                                          message, then the agent takes one turn, its actions
//	                                                          included
//	GET  /ap/v1/agent/tasks/{task_id}/steps                   the steps of a task
//	GET  /ap/v1/agent/tasks/{task_id}/steps/{step_id}         a step
//	GET  /ap/v1/agent/tasks/{task_id}/artifacts               none: the agent makes no artifacts
//
// A step's output is the text of the turn (a reply or a thought); it is the last step when its strategy says the agent
// is done (end_turn), otherwise the next step reasons on what it added. Tasks are kept in memory, a task's steps one at a time. Lists are not paginated: all
// the items are on page 1.
package agentprotocolhttp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/google/uuid"

	"github.com/morebec/proxias/agentsdk"
)

// MaxRequestBytes bounds the body of a request.
const MaxRequestBytes = 1 << 20

// TaskRequest is the body of a task's creation, and of a step's execution.
type TaskRequest struct {
	Input           string         `json:"input,omitempty"`
	AdditionalInput map[string]any `json:"additional_input,omitempty"`
}

// Task is a task of the protocol.
type Task struct {
	TaskID          string         `json:"task_id"`
	Input           string         `json:"input,omitempty"`
	AdditionalInput map[string]any `json:"additional_input,omitempty"`
	Artifacts       []any          `json:"artifacts"`
}

// Step is a step of the protocol: a turn of the agent.
type Step struct {
	TaskID           string         `json:"task_id"`
	StepID           string         `json:"step_id"`
	Name             string         `json:"name"`
	Status           string         `json:"status"`
	Input            string         `json:"input,omitempty"`
	AdditionalInput  map[string]any `json:"additional_input,omitempty"`
	Output           string         `json:"output"`
	AdditionalOutput map[string]any `json:"additional_output,omitempty"`
	Artifacts        []any          `json:"artifacts"`
	IsLast           bool           `json:"is_last"`
}

// task is a task, its session and its steps, and its lock: one step at a time.
type task struct {
	sync.Mutex
	Task
	session agentsdk.Session
	steps   []Step
}

// handler serves an agent.
type handler struct {
	agent    agentsdk.Agent
	strategy agentsdk.CognitiveStrategy

	mutex sync.Mutex
	tasks []*task
}

// NewHandler returns the handler of the agent, whose steps are turns with the strategy.
func NewHandler(agent agentsdk.Agent, strategy agentsdk.CognitiveStrategy) http.Handler {
	h := &handler{agent: agent, strategy: strategy}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /ap/v1/agent/tasks", h.createTask)
	mux.HandleFunc("GET /ap/v1/agent/tasks", h.listTasks)
	mux.HandleFunc("GET /ap/v1/agent/tasks/{task_id}", h.getTask)
	mux.HandleFunc("POST /ap/v1/agent/tasks/{task_id}/steps", h.executeStep)
	mux.HandleFunc("GET /ap/v1/agent/tasks/{task_id}/steps", h.listSteps)
	mux.HandleFunc("GET /ap/v1/agent/tasks/{task_id}/steps/{step_id}", h.getStep)
	mux.HandleFunc("GET /ap/v1/agent/tasks/{task_id}/artifacts", h.listArtifacts)
	return mux
}

func (h *handler) createTask(w http.ResponseWriter, r *http.Request) {
	req, ok := h.read(w, r)
	if !ok {
		return
	}
	t := &task{Task: Task{TaskID: uuid.NewString(), Input: req.Input, AdditionalInput: req.AdditionalInput, Artifacts: []any{}}}
	t.session = agentsdk.NewSessionWithID(t.TaskID)
	if req.Input != "" {
		t.session = t.session.WithEvents(agentsdk.NewMessageEvent(agentsdk.NewUserSource("user"), req.Input))
	}
	h.mutex.Lock()
	h.tasks = append(h.tasks, t)
	h.mutex.Unlock()
	h.write(w, http.StatusOK, t.Task)
}

func (h *handler) listTasks(w http.ResponseWriter, _ *http.Request) {
	h.mutex.Lock()
	tasks := make([]Task, 0, len(h.tasks))
	for _, t := range h.tasks {
		tasks = append(tasks, t.Task)
	}
	h.mutex.Unlock()
	h.write(w, http.StatusOK, map[string]any{"tasks": tasks, "pagination": h.page(len(tasks))})
}

func (h *handler) getTask(w http.ResponseWriter, r *http.Request) {
	if t, ok := h.task(w, r); ok {
		h.write(w, http.StatusOK, t.Task)
	}
}

func (h *handler) executeStep(w http.ResponseWriter, r *http.Request) {
	t, ok := h.task(w, r)
	if !ok {
		return
	}
	req, ok := h.read(w, r)
	if !ok {
		return
	}
	t.Lock()
	defer t.Unlock()

	s := t.session
	if req.Input != "" {
		s = s.WithEvents(agentsdk.NewMessageEvent(agentsdk.NewUserSource("user"), req.Input))
	}
	stream, err := h.agent.Invoke(r.Context(), s, h.strategy)
	if err != nil {
		h.fail(w, http.StatusInternalServerError, err.Error())
		return
	}
	ended, err := stream.Collect()
	if err != nil {
		h.fail(w, http.StatusInternalServerError, err.Error())
		return
	}
	added := ended.Events()[len(s.Events()):]
	s = ended
	// A step is the last when its strategy says the agent is done (end_turn).
	step := Step{TaskID: t.TaskID, StepID: uuid.NewString(), Name: fmt.Sprintf("turn %d", len(t.steps)+1), Status: "completed",
		Input: req.Input, AdditionalInput: req.AdditionalInput, Artifacts: []any{}, IsLast: stream.StopReason() == agentsdk.StopReasonEndTurn}
	actions := 0
	for _, e := range added {
		if (e.Type == agentsdk.EventTypeMessage || e.Type == agentsdk.EventTypeThought) && e.Source.Type == agentsdk.EventSourceTypeAgent && e.Text() != "" {
			step.Output = e.Text()
		}
		if e.Type == agentsdk.EventTypeActionResult {
			actions++
		}
	}
	if actions > 0 {
		step.AdditionalOutput = map[string]any{"actions": actions}
	}
	t.session = s
	t.steps = append(t.steps, step)
	h.write(w, http.StatusOK, step)
}

func (h *handler) listSteps(w http.ResponseWriter, r *http.Request) {
	t, ok := h.task(w, r)
	if !ok {
		return
	}
	t.Lock()
	steps := append([]Step{}, t.steps...)
	t.Unlock()
	h.write(w, http.StatusOK, map[string]any{"steps": steps, "pagination": h.page(len(steps))})
}

func (h *handler) getStep(w http.ResponseWriter, r *http.Request) {
	t, ok := h.task(w, r)
	if !ok {
		return
	}
	t.Lock()
	defer t.Unlock()
	for _, step := range t.steps {
		if step.StepID == r.PathValue("step_id") {
			h.write(w, http.StatusOK, step)
			return
		}
	}
	h.fail(w, http.StatusNotFound, "no step "+r.PathValue("step_id"))
}

func (h *handler) listArtifacts(w http.ResponseWriter, r *http.Request) {
	if _, ok := h.task(w, r); ok {
		h.write(w, http.StatusOK, map[string]any{"artifacts": []any{}, "pagination": h.page(0)})
	}
}

// task is the task of the request's task_id; a missing one is answered 404.
func (h *handler) task(w http.ResponseWriter, r *http.Request) (*task, bool) {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	for _, t := range h.tasks {
		if t.TaskID == r.PathValue("task_id") {
			return t, true
		}
	}
	h.fail(w, http.StatusNotFound, "no task "+r.PathValue("task_id"))
	return nil, false
}

// read reads the body of the request, if any; one that is not a TaskRequest is answered 400.
func (h *handler) read(w http.ResponseWriter, r *http.Request) (TaskRequest, bool) {
	var req TaskRequest
	if r.ContentLength == 0 {
		return req, true
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, MaxRequestBytes)).Decode(&req); err != nil {
		h.fail(w, http.StatusBadRequest, "the body is not a task request")
		return req, false
	}
	return req, true
}

// page is the pagination of a list of the items, all on one page.
func (h *handler) page(items int) map[string]int {
	return map[string]int{"total_items": items, "total_pages": 1, "current_page": 1, "page_size": max(items, 1)}
}

func (h *handler) fail(w http.ResponseWriter, status int, message string) {
	h.write(w, status, map[string]string{"message": message})
}

func (h *handler) write(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v) // a client gone is told by the context
}
