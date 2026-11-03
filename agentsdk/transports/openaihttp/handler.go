// Package openaihttp serves agents through the OpenAI Chat Completions API, so
// that any OpenAI client talks to them as to models: an agent's ID is a model.
//
//	GET  /v1/models            the agents
//	POST /v1/chat/completions  the agent of the model Loops on the request's messages; the reply as a chat.completion,
//	                           or as chat.completion.chunk events (SSE) ending with "data: [DONE]" when stream is true
//
// It is stateless: the client sends the history. System and developer messages
// are instructions (messages of a system source), user messages a person's,
// assistant messages the agent's own; tool calls and tool results are the
// client's, not the agent's actions, and are ignored. Usage is not reported:
// agentsdk does not count tokens.
package openaihttp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"

	"github.com/morebec/proxias/agentsdk"
)

// MaxRequestBytes bounds the body of a request.
const MaxRequestBytes = 1 << 20

// ChatCompletionRequest is the part of a request of the Chat Completions API
// that the handler reads.
type ChatCompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

// Message is a message of a request: its content is a text, or text parts.
type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
	Name    string          `json:"name,omitempty"`
}

// Text is the text of the content: the text, or its text parts, one after the
// other.
func (m Message) Text() string {
	var text string
	if json.Unmarshal(m.Content, &text) == nil {
		return text
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	_ = json.Unmarshal(m.Content, &parts) // content that is neither is no text
	for _, p := range parts {
		if p.Type == "text" {
			text += p.Text
		}
	}
	return text
}

// handler serves the agents.
type handler struct {
	agents   map[string]agentsdk.Agent
	order    []string
	strategy agentsdk.CognitiveStrategy
}

// NewHandler returns the handler of the agents, which Loop with the strategy on
// each request.
func NewHandler(strategy agentsdk.CognitiveStrategy, agents []agentsdk.Agent) http.Handler {
	h := &handler{agents: map[string]agentsdk.Agent{}, strategy: strategy}
	for _, a := range agents {
		h.agents[a.ID] = a
		h.order = append(h.order, a.ID)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", h.models)
	mux.HandleFunc("POST /v1/chat/completions", h.complete)
	return mux
}

func (h *handler) models(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	models := make([]model, 0, len(h.order))
	for _, id := range h.order {
		models = append(
			models,
			model{ID: id, Object: "model", OwnedBy: "agentsdk"},
		)
	}
	h.write(w, http.StatusOK, map[string]any{"object": "list", "data": models})
}

func (h *handler) complete(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	body := http.MaxBytesReader(w, r.Body, MaxRequestBytes)
	if err := json.NewDecoder(body).Decode(&req); err != nil {
		h.fail(w, http.StatusBadRequest, "invalid_request_error", "the body is not a chat completion request")
		return
	}
	agent, ok := h.agents[req.Model]
	if !ok {
		h.fail(w, http.StatusNotFound, "model_not_found", fmt.Sprintf("the model %q does not exist", req.Model))
		return
	}
	if len(req.Messages) == 0 {
		h.fail(w, http.StatusBadRequest, "invalid_request_error", "messages is empty")
		return
	}

	s := h.session(agent, req.Messages)
	c := completion{
		ID:      "chatcmpl-" + uuid.NewString(),
		Created: time.Now().Unix(),
		Model:   req.Model,
	}
	updates, err := agent.Loop(r.Context(), s, h.strategy)
	if err != nil {
		h.fail(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	if !req.Stream {
		ended, err := updates.Collect()
		if err != nil {
			h.fail(w, http.StatusInternalServerError, "server_error", err.Error())
			return
		}
		finish := "stop"
		if updates.StopReason() == agentsdk.StopReasonMaxTurns || updates.StopReason() == agentsdk.StopReasonLength {
			// The agent took all its turns, or its answer was cut.
			finish = "length"
		}
		if updates.StopReason() == agentsdk.StopReasonContentFilter {
			finish = "content_filter"
		}
		h.write(w, http.StatusOK, c.whole(ended.Events()[len(s.Events()):], finish))
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)
	send := func(v any) {
		data, _ := json.Marshal(v) // the chunks marshal
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	}
	send(c.chunk(map[string]any{"role": "assistant", "content": ""}, nil))
	// A text's chunks are held until its event comes, and sent only when it is
	// a reply: a text the model followed
	// with tool calls is a thought.
	held := map[agentsdk.EventID][]string{}
	for d := range updates.Updates() {
		if d.Err != nil {
			send(
				map[string]any{
					"error": apiError{
						Message: d.Err.Error(),
						Type:    "server_error",
					},
				},
			)
			break
		}
		if d.Chunk != nil {
			held[d.Chunk.EventID] = append(held[d.Chunk.EventID], string(d.Chunk.Data))
		}
		for _, e := range d.Events {
			if e.Type != agentsdk.EventTypeMessage || e.Source != agent.AsSource() {
				continue
			}
			texts, ok := held[e.ID]
			if !ok && e.Text() != "" {
				texts = []string{e.Text()}
			}
			for _, text := range texts {
				send(c.chunk(map[string]any{"content": text}, nil))
			}
		}
	}
	stop := "stop"
	if updates.StopReason() == agentsdk.StopReasonMaxTurns || updates.StopReason() == agentsdk.StopReasonLength {
		stop = "length" // the agent took all its turns, or its answer was cut
	}
	if updates.StopReason() == agentsdk.StopReasonContentFilter {
		stop = "content_filter"
	}
	send(c.chunk(map[string]any{}, &stop))
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
}

// session is the messages as a session of the agent.
func (h *handler) session(agent agentsdk.Agent, messages []Message) agentsdk.Session {
	s := agentsdk.NewSession()
	for _, m := range messages {
		text := m.Text()
		if text == "" {
			continue
		}
		switch m.Role {
		case "system", "developer":
			s = s.WithEvents(agentsdk.NewMessageEvent(agentsdk.NewSystemSource("instructions"), text))
		case "user":
			name := m.Name
			if name == "" {
				name = "user"
			}
			s = s.WithEvents(agentsdk.NewMessageEvent(agentsdk.NewUserSource(name), text))
		case "assistant":
			s = s.WithEvents(agentsdk.NewMessageEvent(agent.AsSource(), text))
		}
	}
	return s
}

// apiError is an error in OpenAI's shape.
type apiError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

func (h *handler) fail(w http.ResponseWriter, status int, kind, message string) {
	h.write(
		w,
		status,
		map[string]any{"error": apiError{Message: message, Type: kind}},
	)
}

func (h *handler) write(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v) // a client gone is told by the context
}

// completion is what the chat completion of a request is: its ID, date and
// model.
type completion struct {
	ID      string
	Created int64
	Model   string
}

// whole is the chat.completion of the events, the text of the last message of
// an agent, ended for the finish reason.
func (c completion) whole(events []agentsdk.Event, finish string) map[string]any {
	reply := ""
	for _, e := range events {
		if e.Type == agentsdk.EventTypeMessage && e.HasSourceType(agentsdk.EventSourceTypeAgent) {
			reply = e.Text()
		}
	}
	return map[string]any{
		"id":      c.ID,
		"object":  "chat.completion",
		"created": c.Created,
		"model":   c.Model,
		"choices": []map[string]any{{
			"index": 0,
			"message": map[string]any{
				"role":    "assistant",
				"content": reply,
			},
			"finish_reason": finish,
		}},
	}
}

// chunk is a chat.completion.chunk of the delta.
func (c completion) chunk(delta map[string]any, finish *string) map[string]any {
	return map[string]any{
		"id":      c.ID,
		"object":  "chat.completion.chunk",
		"created": c.Created,
		"model":   c.Model,
		"choices": []map[string]any{
			{"index": 0, "delta": delta, "finish_reason": finish},
		},
	}
}
