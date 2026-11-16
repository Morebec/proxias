package proxias

import (
	"context"
	"iter"
	"sync"
)

// StopReason is why the agent stopped, as a model's stop reason.
type StopReason string

const (
	// StopReasonEndTurn is an agent done: it replied, or has nothing to add.
	// Loop
	// stops. in other words, it is a natural end, or a stop sequence.
	StopReasonEndTurn StopReason = "end_turn"

	// StopReasonMaxTurns is a Loop that took all its turns without one ending
	// with
	// StopReasonEndTurn. It is not an error: the agent stopped where it was.
	StopReasonMaxTurns StopReason = "max_turns"

	// StopReasonLength is an agent whose answer was cut at the limit of tokens.
	// Loop
	// stops.
	StopReasonLength StopReason = "length"

	// StopReasonContentFilter is an agent whose model declined to answer. Loop
	// stops.
	StopReasonContentFilter StopReason = "content_filter"
)

// SessionStream is a call of an agent (Invoke, Loop or Watch): its session,
// which each update sent grows, and the updates, read with Updates or Collect.
// The session given to the call only seeds it.
type SessionStream struct {
	mutex   sync.Mutex
	session Session
	stop    StopReason
	updates chan StreamUpdate
}

// Session is the session so far: the one the call was given, with the events of
// every update sent. It may be ahead of the updates read; once they are read to
// their end, it is the session the call ended with.
func (s *SessionStream) Session() Session {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return s.session
}

// StopReason is the last StopReason sent, "" since an error: once the updates
// are read to their end, how the call ended (StopReasonEndTurn,
// StopReasonMaxTurns, or "" for an error or a context that ended).
func (s *SessionStream) StopReason() StopReason {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return s.stop
}

// Updates reads the updates as they come. An update that carries an Err is the
// error that ended the call, last, or, for a Watch, one of the loop on an
// event, after which the next event still comes. Stopping early leaves the call
// running
// until its context ends.
func (s *SessionStream) Updates() iter.Seq[StreamUpdate] {
	return func(yield func(StreamUpdate) bool) {
		for u := range s.updates {
			if !yield(u) {
				return
			}
		}
	}
}

// Collect reads the updates to their end, for the callers that want only the
// result: the session the call ended with, and its first error. The events the
// call added are those after the session it was given.
func (s *SessionStream) Collect() (Session, error) {
	var first error
	for u := range s.Updates() {
		if u.Err != nil && first == nil {
			first = u.Err
		}
	}
	return s.Session(), first
}

// close ends the stream: Updates ends once the updates sent are read.
func (s *SessionStream) close() { close(s.updates) }

// send adds the events of the update to the session, then sends the update; it
// fails with ctx's error when ctx ends first.
func (s *SessionStream) send(ctx context.Context, u StreamUpdate) error {
	s.mutex.Lock()
	s.session = s.session.WithEvents(u.Events...)
	switch {
	case u.Err != nil:
		s.stop = ""
	case u.StopReason != "":
		s.stop = u.StopReason
	}
	s.mutex.Unlock()
	select {
	case s.updates <- u:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// StreamUpdate is what a step adds: whole events, which enter the session, or a
// chunk of an event still being produced, which does not; or the error of the
// turn (Err). When the agent stops, the last update says why (StopReason).
type StreamUpdate struct {
	Events []Event
	Chunk  *StreamChunk
	Err    error
	// StopReason is set only when the agent stops, on the last update: by the
	// strategy when the agent is done, by Loop when it took its turns. A turn
	// without one goes on: Loop takes another.
	StopReason StopReason
}

// StreamChunk is a piece of the text of an event still being produced, as the
// model writes it.
type StreamChunk struct {
	// EventID is the event the chunk is a piece of.
	EventID EventID
	Data    []byte
	// Thinking tells that the chunk is a piece of the model's thinking as it
	// writes it, before the event, rather than of the event's text: it is only
	// to be shown, the thinking is kept whole with the event.
	Thinking bool
}
