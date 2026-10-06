package proxias

import (
	"context"
	"errors"
)

// ErrNoFallback is returned by Intercept for a message that needs a Fallback when there is none.
var ErrNoFallback = errors.New("proxias: no fallback is configured")

// Fallback answers a message the Interceptor did not (entirely) handle, usually by asking an LLM. It gets
// the Results of the Workflows run for the message, none when there was no Intent. Its Completion is
// whatever the host needs to answer (a stream, an HTTP response...), opaque to Proxias.
type Fallback interface {
	Complete(ctx context.Context, request Request, results []Result) (any, error)
}

// FallbackFunc makes a Fallback of a function.
type FallbackFunc func(ctx context.Context, request Request, results []Result) (any, error)

func (f FallbackFunc) Complete(ctx context.Context, request Request, results []Result) (any, error) {
	return f(ctx, request, results)
}
