package agenthttp_test

import (
	"bufio"
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

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/transports/agenthttp"
)

func TestHandlerPostMessage(t *testing.T) {
	tests := []struct {
		name  string
		when  string
		code  int
		lines []string
	}{
		{
			name: "WHEN a message THEN the message, the chunk and the reply stream as NDJSON",
			when: `{"from":"john","text":"hello"}`,
			code: http.StatusOK,
			lines: []string{
				"event user:john hello",
				"chunk Hi.",
				"event agent:ada Hi.",
			},
		},
		{
			name: "WHEN not JSON THEN it is a bad request",
			when: "hello",
			code: http.StatusBadRequest,
		},
		{
			name: "WHEN no text THEN it is a bad request",
			when: `{"from":"john"}`,
			code: http.StatusBadRequest,
		},
		{
			name: "WHEN too large THEN it is rejected",
			when: `{"from":"john","text":"` + strings.Repeat(
				"a",
				agenthttp.MaxMessageBytes,
			) + `"}`,
			code: http.StatusRequestEntityTooLarge,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := proxias.StrategyFunc(
				func(context.Context, proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
					return func(yield func(proxias.StreamUpdate) bool) {
						if !yield(
							proxias.StreamUpdate{
								Chunk: &proxias.StreamChunk{
									EventID: "r",
									Data:    []byte("Hi."),
								},
							},
						) {
							return
						}
						yield(
							proxias.StreamUpdate{
								Events: []proxias.Event{
									{
										ID:      "r",
										Type:    proxias.EventTypeMessage,
										Source:  proxias.NewAgentSource("ada"),
										Payload: []byte("Hi."),
										Date:    time.Now(),
									},
								},
								StopReason: proxias.StopReasonEndTurn,
							},
						)
					}
				},
			)
			handler := agenthttp.NewHandler(proxias.Agent{ID: "ada"}, strategy)

			w := httptest.NewRecorder()
			handler.ServeHTTP(
				w,
				httptest.NewRequestWithContext(
					t.Context(),
					http.MethodPost,
					"/sessions/s1/messages",
					strings.NewReader(tt.when),
				),
			)

			assert.Equal(t, tt.code, w.Code)
			var lines []string
			scanner := bufio.NewScanner(w.Body)
			for tt.code == http.StatusOK && scanner.Scan() {
				var line agenthttp.Update
				require.NoError(t, json.Unmarshal(scanner.Bytes(), &line))
				if line.Chunk != nil {
					lines = append(lines, "chunk "+string(line.Chunk.Data))
				}
				for _, e := range line.Events {
					lines = append(lines, "event "+e.Source.String()+" "+e.Text())
				}
			}
			assert.Equal(t, tt.lines, lines)
		})
	}
}

func TestHandlerGetSession(t *testing.T) {
	tests := []struct {
		name   string
		given  []string
		events int
	}{
		{name: "GIVEN no message THEN the session is empty", events: 0},
		{
			name:   "GIVEN messages THEN the session keeps them and the replies",
			given:  []string{"hello", "again"},
			events: 4,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := proxias.StrategyFunc(
				func(context.Context, proxias.Invocation) iter.Seq[proxias.StreamUpdate] {
					return func(yield func(proxias.StreamUpdate) bool) {
						yield(
							proxias.StreamUpdate{
								Events: []proxias.Event{
									proxias.NewEvent(
										proxias.EventTypeMessage,
										proxias.NewAgentSource("ada"),
										[]byte("Hi."),
									),
								},
								StopReason: proxias.StopReasonEndTurn,
							},
						)
					}
				},
			)
			handler := agenthttp.NewHandler(proxias.Agent{ID: "ada"}, strategy)
			for _, text := range tt.given {
				w := httptest.NewRecorder()
				handler.ServeHTTP(
					w,
					httptest.NewRequestWithContext(
						t.Context(),
						http.MethodPost,
						"/sessions/s1/messages",
						strings.NewReader(`{"from":"john","text":"`+text+`"}`),
					),
				)
				require.Equal(t, http.StatusOK, w.Code)
			}

			w := httptest.NewRecorder()
			handler.ServeHTTP(w, httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/sessions/s1", nil))

			var session struct {
				SessionID string          `json:"sessionId"`
				Events    []proxias.Event `json:"events"`
			}
			require.NoError(t, json.NewDecoder(w.Body).Decode(&session))
			assert.Equal(t, "s1", session.SessionID)
			assert.Len(t, session.Events, tt.events)
		})
	}
}
