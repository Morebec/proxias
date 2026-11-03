// Package agenthttp serves an agent over HTTP: one transport among any (a WebSocket, gRPC, a queue would be others),
// which turns its requests into events for the agent and the agent's stream into its responses.
//
//	POST /sessions/{id}/messages  {"from": "john", "text": "..."}  the updates of the Loop, as NDJSON, as they come
//	GET  /sessions/{id}                                            the session, as JSON
//
// Sessions are kept in memory, a session's loops one at a time.
package agenthttp

import (
	"encoding/json"
	"errors"
	"net/http"
	"sync"

	"github.com/morebec/proxias/agentsdk"
)

// MaxMessageBytes bounds the body of a message.
const MaxMessageBytes = 1 << 20

// Message is the body of a message to the agent.
type Message struct {
	// From is the ID of the person who writes it.
	From string `json:"from"`
	Text string `json:"text"`
}

// Update is a line of the stream: whole events, or a chunk of one; the last line carries the error, if any.
type Update struct {
	Events []agentsdk.Event      `json:"events,omitempty"`
	Chunk  *agentsdk.StreamChunk `json:"chunk,omitempty"`
	Error  string                `json:"error,omitempty"`
}

// handler serves an agent.
type handler struct {
	agent    agentsdk.Agent
	strategy agentsdk.CognitiveStrategy

	mutex    sync.Mutex
	sessions map[string]*session
}

// session is a session and its lock: one loop at a time.
type session struct {
	sync.Mutex
	agentsdk.Session
}

// NewHandler returns the handler of the agent, which Loops with the strategy on each message.
func NewHandler(agent agentsdk.Agent, strategy agentsdk.CognitiveStrategy) http.Handler {
	h := &handler{agent: agent, strategy: strategy, sessions: map[string]*session{}}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /sessions/{id}/messages", h.postMessage)
	mux.HandleFunc("GET /sessions/{id}", h.getSession)
	return mux
}

// session is the session of the ID, made when there is none.
func (h *handler) session(id string) *session {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	s, ok := h.sessions[id]
	if !ok {
		s = &session{Session: agentsdk.NewSessionWithID(id)}
		h.sessions[id] = s
	}
	return s
}

func (h *handler) postMessage(w http.ResponseWriter, r *http.Request) {
	var message Message
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, MaxMessageBytes)).Decode(&message); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			http.Error(w, "the message is too large", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "the body is not a message", http.StatusBadRequest)
		return
	}
	if message.From == "" || message.Text == "" {
		http.Error(w, `a message needs "from" and "text"`, http.StatusBadRequest)
		return
	}

	s := h.session(r.PathValue("id"))
	s.Lock()
	defer s.Unlock()

	w.Header().Set("Content-Type", "application/x-ndjson")
	encoder := json.NewEncoder(w)
	flusher, _ := w.(http.Flusher)
	event := agentsdk.NewMessageEvent(agentsdk.NewUserSource(message.From), message.Text)
	s.Session = s.WithEvents(event)
	_ = encoder.Encode(Update{Events: []agentsdk.Event{s.LastEvent()}}) // a client gone is told by the context, which ends the loop
	stream, err := h.agent.Loop(r.Context(), s.Session, h.strategy)
	if err != nil {
		_ = encoder.Encode(Update{Error: err.Error()})
		return
	}
	for d := range stream.Updates() {
		line := Update{Events: d.Events, Chunk: d.Chunk}
		if d.Err != nil {
			line.Error = d.Err.Error()
		}
		_ = encoder.Encode(line)
		if flusher != nil {
			flusher.Flush()
		}
	}
	s.Session = stream.Session()
}

func (h *handler) getSession(w http.ResponseWriter, r *http.Request) {
	s := h.session(r.PathValue("id"))
	s.Lock()
	defer s.Unlock()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"sessionId": s.ID(),
		"events":    s.Events(),
	})
}
