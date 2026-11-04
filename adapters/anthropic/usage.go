package anthropic

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go/option"
)

// Usage is what the calls to Claude cost: how many, and their tokens.
type Usage struct {
	Calls                    int
	InputTokens              int64
	OutputTokens             int64
	CacheCreationInputTokens int64
	CacheReadInputTokens     int64
	// CallDuration is the time the calls took, from their request to their
	// response read whole.
	CallDuration time.Duration
}

// UsageCounter counts the usage of the calls of the clients built with its
// Middleware: of every component that uses them (a ReAct shell's Model as much
// as a classifier or a synthesizer of Proxias). It reads the usage of the
// responses that are JSON, and of those streamed, once their stream is read.
// It is safe for concurrent use.
type UsageCounter struct {
	mutex sync.Mutex
	usage Usage
}

// tokens is the usage of a response, as Claude tells it.
type tokens struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
}

// Middleware is the client option that counts its calls:
// option.WithMiddleware(counter.Middleware()).
func (c *UsageCounter) Middleware() option.Middleware {
	return func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		began := time.Now()
		resp, err := next(req)
		if err != nil {
			return resp, err
		}
		contentType := resp.Header.Get("Content-Type")
		if strings.HasPrefix(contentType, "text/event-stream") && resp.Body != nil {
			resp.Body = &countedStream{
				body:    resp.Body,
				counter: c,
				began:   began,
			}
			return resp, nil
		}

		var usage struct {
			Usage tokens `json:"usage"`
		}
		if strings.HasPrefix(contentType, "application/json") && resp.Body != nil {
			body, rerr := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			resp.Body = io.NopCloser(bytes.NewReader(body))
			if rerr != nil {
				return resp, rerr
			}
			_ = json.Unmarshal(body, &usage) // a body without usage counts none
		}
		c.count(usage.Usage, time.Since(began))
		return resp, nil
	}
}

// count adds a call and its tokens.
func (c *UsageCounter) count(spent tokens, took time.Duration) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.usage.Calls++
	c.usage.InputTokens += spent.InputTokens
	c.usage.OutputTokens += spent.OutputTokens
	c.usage.CacheCreationInputTokens += spent.CacheCreationInputTokens
	c.usage.CacheReadInputTokens += spent.CacheReadInputTokens
	c.usage.CallDuration += took
}

// countedStream is a streamed response, counted once it is read to its end or
// closed: its input tokens are in its message_start event, its output tokens
// in its message_delta events, the last one the total.
type countedStream struct {
	body    io.ReadCloser
	counter *UsageCounter
	began   time.Time
	read    bytes.Buffer
	once    sync.Once
}

func (s *countedStream) Read(p []byte) (int, error) {
	n, err := s.body.Read(p)
	s.read.Write(p[:n])
	if err != nil {
		s.end()
	}
	return n, err
}

func (s *countedStream) Close() error {
	s.end()
	return s.body.Close()
}

// end counts the stream, once.
func (s *countedStream) end() {
	s.once.Do(func() {
		var spent tokens
		scanner := bufio.NewScanner(bytes.NewReader(s.read.Bytes()))
		scanner.Buffer(make([]byte, 64<<10), 4<<20)
		for scanner.Scan() {
			data, ok := bytes.CutPrefix(scanner.Bytes(), []byte("data: "))
			if !ok {
				continue
			}
			var event struct {
				Type    string `json:"type"`
				Message struct {
					Usage tokens `json:"usage"`
				} `json:"message"`
				Usage tokens `json:"usage"`
			}
			if json.Unmarshal(data, &event) != nil {
				continue
			}
			switch event.Type {
			case "message_start":
				spent = event.Message.Usage
			case "message_delta":
				spent.OutputTokens = event.Usage.OutputTokens
			}
		}
		s.counter.count(spent, time.Since(s.began))
	})
}

// Usage is the usage counted so far.
func (c *UsageCounter) Usage() Usage {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.usage
}

// Reset returns the usage counted so far, and counts from zero again.
func (c *UsageCounter) Reset() Usage {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	usage := c.usage
	c.usage = Usage{}
	return usage
}
