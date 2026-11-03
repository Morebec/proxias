package anthropic

import (
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
// responses that are JSON; a streamed response is counted as a call, without
// its tokens. It is safe for
// concurrent use.
type UsageCounter struct {
	mutex sync.Mutex
	usage Usage
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
		var usage struct {
			Usage struct {
				InputTokens              int64 `json:"input_tokens"`
				OutputTokens             int64 `json:"output_tokens"`
				CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
			} `json:"usage"`
		}
		if strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") && resp.Body != nil {
			body, rerr := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			resp.Body = io.NopCloser(bytes.NewReader(body))
			if rerr != nil {
				return resp, rerr
			}
			_ = json.Unmarshal(body, &usage) // a body without usage counts none
		}

		c.mutex.Lock()
		defer c.mutex.Unlock()
		c.usage.Calls++
		c.usage.InputTokens += usage.Usage.InputTokens
		c.usage.OutputTokens += usage.Usage.OutputTokens
		c.usage.CacheCreationInputTokens += usage.Usage.CacheCreationInputTokens
		c.usage.CacheReadInputTokens += usage.Usage.CacheReadInputTokens
		c.usage.CallDuration += time.Since(began)
		return resp, nil
	}
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
