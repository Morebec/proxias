package proxias_test

import (
	"context"
	"errors"
	"iter"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

// memory is a MemoryProvider of a function.
type memory func(proxias.Invocation) (string, error)

func (m memory) Recall(
	_ context.Context,
	inv proxias.Invocation,
) (string, error) {
	return m(inv)
}

func TestAgentMemory(t *testing.T) {
	tests := []struct {
		name string
		// memory is the agent's: none when nil.
		memory memory
		// told is what the agent's prompt tells, missing what it must not;
		// then the events of the stream, err its error.
		told, missing string
		then          []string
		err           string
	}{
		{
			name: "GIVEN a memory THEN the turn is told what it recalls, " +
				"not in its session",
			memory: func(proxias.Invocation) (string, error) {
				return "the demo is on friday", nil
			},
			told: "the demo is on friday",
			then: []string{"when is the demo?", "Friday."},
		},
		{
			name: "GIVEN a memory that recalls nothing " +
				"THEN the turn is told no memory",
			memory: func(proxias.Invocation) (string, error) {
				return "", nil
			},
			missing: "What you remember",
			then:    []string{"when is the demo?", "Friday."},
		},
		{
			name:    "GIVEN no memory THEN the turn is told no memory",
			missing: "What you remember",
			then:    []string{"when is the demo?", "Friday."},
		},
		{
			name: "GIVEN the memory fails to recall " +
				"THEN the error ends the turn, before the strategy",
			memory: func(proxias.Invocation) (string, error) {
				return "", errors.New("disk")
			},
			then: []string{"when is the demo?"},
			err:  "recalling: disk",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var told string
			agent := proxias.Agent{ID: "ada"}
			if tt.memory != nil {
				agent.Memory = tt.memory
			}
			strategy := proxias.StrategyFunc(
				func(
					ctx context.Context,
					inv proxias.Invocation,
				) iter.Seq[proxias.StreamUpdate] {
					return func(yield func(proxias.StreamUpdate) bool) {
						agentPrompt, err := proxias.NewAgentPrompt(ctx, inv)
						if err != nil {
							yield(proxias.StreamUpdate{Err: err})
							return
						}
						told = agentPrompt.String()
						yield(proxias.StreamUpdate{
							Events: []proxias.Event{
								proxias.NewMessageEvent(
									proxias.NewAgentSource("ada"),
									"Friday.",
								),
							},
							StopReason: proxias.StopReasonEndTurn,
						})
					}
				},
			)

			stream, err := agent.Invoke(
				context.Background(),
				proxias.NewSession(proxias.NewMessageEvent(
					proxias.NewUserSource("john"),
					"when is the demo?",
				)),
				strategy,
			)
			require.NoError(t, err)
			session, err := stream.Collect()

			if tt.err != "" {
				require.EqualError(t, err, tt.err)
			} else {
				require.NoError(t, err)
			}
			then := make([]string, 0, len(session.Events()))
			for _, e := range session.Events() {
				then = append(then, e.Text())
			}
			assert.Equal(t, tt.then, then)
			if tt.told != "" {
				assert.Contains(t, told, tt.told)
			}
			if tt.missing != "" {
				assert.NotContains(t, told, tt.missing)
			}
		})
	}
}

func TestMarkdownMemory(t *testing.T) {
	tests := []struct {
		name string
		// given is the document, none when ""; recalled what is recalled.
		given    string
		recalled string
	}{
		{
			name: "GIVEN no document THEN nothing is recalled",
		},
		{
			name:     "GIVEN a document THEN it is recalled whole",
			given:    "# John\n- likes tea",
			recalled: "# John\n- likes tea",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "MEMORY.md")
			if tt.given != "" {
				require.NoError(t, os.WriteFile(path, []byte(tt.given), 0o600))
			}
			memory := proxias.MarkdownMemory{Path: path}

			recalled, err := memory.Recall(
				context.Background(),
				proxias.Invocation{},
			)

			require.NoError(t, err)
			assert.Equal(t, tt.recalled, recalled)
		})
	}
}
