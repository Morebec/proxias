package openai_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/adapters/openai"
)

type controlPlaneFunc func(context.Context, proxias.CompletionRequest) (proxias.CompletionResponse, error)

func (f controlPlaneFunc) Process(ctx context.Context, req proxias.CompletionRequest) (proxias.CompletionResponse, error) {
	return f(ctx, req)
}

// spyBody is a response body that remembers whether it was closed.
type spyBody struct {
	io.Reader
	closed bool
}

func (b *spyBody) Close() error { b.closed = true; return nil }

// chunks is a Reader that returns one chunk per Read, as a stream does.
type chunks []string

func (c *chunks) Read(p []byte) (int, error) {
	if len(*c) == 0 {
		return 0, io.EOF
	}
	n := copy(p, (*c)[0])
	*c = (*c)[1:]
	return n, nil
}

// zeroes is a Reader of n zero bytes.
type zeroes int

func (z *zeroes) Read(p []byte) (int, error) {
	if *z <= 0 {
		return 0, io.EOF
	}
	n := min(len(p), int(*z))
	clear(p[:n])
	*z -= zeroes(n)
	return n, nil
}

// completionRequestBody is a chat completion request the handler accepts.
const completionRequestBody = `{"model":"m","messages":[{"role":"user","content":"restart the api"}]}`

// unaskedControlPlane panics if it is asked: it is how a request that must not be processed says so.
var unaskedControlPlane = controlPlaneFunc(func(context.Context, proxias.CompletionRequest) (proxias.CompletionResponse, error) {
	panic("the control plane must not be asked about a request the handler does not accept")
})

// controlPlaneAnswering is a control plane that answers response to every request.
func controlPlaneAnswering(response proxias.CompletionResponse) controlPlaneFunc {
	return func(context.Context, proxias.CompletionRequest) (proxias.CompletionResponse, error) {
		return response, nil
	}
}

// responseBody is the Body of a CompletionResponse that holds s.
func responseBody(s string) io.ReadCloser { return io.NopCloser(strings.NewReader(s)) }

// postCompletionRequest posts body to the completions endpoint.
func postCompletionRequest(t *testing.T, handler http.Handler, body io.Reader) *httptest.ResponseRecorder {
	t.Helper()

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/v1/chat/completions", body))

	return rec
}

func TestHandlerWritesTheControlPlanesResponse(t *testing.T) {
	type expect struct {
		status      int
		contentType string
		body        string
	}
	tests := []struct {
		name  string
		given proxias.CompletionResponse // what the control plane answers
		then  expect
	}{
		{
			name: "GIVEN a response with a status, a content type and a body THEN it is written as it is",
			given: proxias.CompletionResponse{
				Status:      http.StatusCreated,
				ContentType: "application/json",
				Body:        responseBody(`{"ok":true}`),
			},
			then: expect{
				status:      http.StatusCreated,
				contentType: "application/json",
				body:        `{"ok":true}`,
			},
		},
		{
			name: "GIVEN a response without a status THEN it is written as a 200",
			given: proxias.CompletionResponse{
				ContentType: "text/plain",
				Body:        responseBody("hi"),
			},
			then: expect{
				status:      http.StatusOK,
				contentType: "text/plain",
				body:        "hi",
			},
		},
		{
			name: "GIVEN a response with a status that is not one THEN it is written as a 500",
			given: proxias.CompletionResponse{
				Status: 1000,
				Body:   responseBody("hi"),
			},
			then: expect{
				status: http.StatusInternalServerError,
				body:   "hi",
			},
		},
		{
			name:  "GIVEN a response without a body THEN only its status is written",
			given: proxias.CompletionResponse{Status: http.StatusNoContent},
			then:  expect{status: http.StatusNoContent},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := openai.Handler(controlPlaneAnswering(tt.given))

			rec := postCompletionRequest(t, handler, strings.NewReader(completionRequestBody))

			assert.Equal(t, tt.then.status, rec.Code)
			assert.Equal(t, tt.then.contentType, rec.Header().Get("Content-Type"))
			assert.Equal(t, tt.then.body, rec.Body.String())
		})
	}
}

func TestHandlerRejectsWhatItCannotProcess(t *testing.T) {
	// The control plane panics if it is asked: none of these requests reaches it.
	handler := openai.Handler(unaskedControlPlane)
	hugeBody := zeroes(33 << 20)

	type args struct {
		method string
		body   io.Reader
	}
	type expect struct {
		status      int
		contentType string
		body        string
		allow       string
	}
	tests := []struct {
		name string
		when args
		then expect
	}{
		{
			name: "WHEN a body that is not a request is posted THEN it is a 400 that says why",
			when: args{
				method: http.MethodPost,
				body:   strings.NewReader(`restart the api`),
			},
			then: expect{
				status:      http.StatusBadRequest,
				contentType: "text/plain; charset=utf-8",
				body:        "openai: decode the request: invalid character 'r' looking for beginning of value\n",
			},
		},
		{
			name: "WHEN a body that is too large is posted THEN it is a 413",
			when: args{
				method: http.MethodPost,
				body:   &hugeBody,
			},
			then: expect{
				status:      http.StatusRequestEntityTooLarge,
				contentType: "text/plain; charset=utf-8",
				body:        "the request is too large\n",
			},
		},
		{
			name: "WHEN a request is not posted THEN it is a 405 that says POST is the one method",
			when: args{
				method: http.MethodGet,
				body:   strings.NewReader(completionRequestBody),
			},
			then: expect{
				status:      http.StatusMethodNotAllowed,
				contentType: "text/plain; charset=utf-8",
				body:        "method not allowed\n",
				allow:       http.MethodPost,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, httptest.NewRequestWithContext(t.Context(), tt.when.method, "/v1/chat/completions", tt.when.body))

			assert.Equal(t, tt.then.status, rec.Code)
			assert.Equal(t, tt.then.contentType, rec.Header().Get("Content-Type"))
			assert.Equal(t, tt.then.allow, rec.Header().Get("Allow"))
			assert.Equal(t, tt.then.body, rec.Body.String())
		})
	}
}

func TestHandlerAnswersABadGatewayWhenTheControlPlaneFails(t *testing.T) {
	// Given a control plane that fails, naming an internal address
	handler := openai.Handler(controlPlaneFunc(func(context.Context, proxias.CompletionRequest) (proxias.CompletionResponse, error) {
		return proxias.CompletionResponse{}, errors.New("the classifier at 10.0.0.7 is down")
	}))

	// When a request is posted
	rec := postCompletionRequest(t, handler, strings.NewReader(completionRequestBody))

	// Then it is a 502 that does not say why
	assert.Equal(t, http.StatusBadGateway, rec.Code)
	assert.Equal(t, "text/plain; charset=utf-8", rec.Header().Get("Content-Type"))
	assert.Equal(t, "the request could not be answered\n", rec.Body.String())
}

func TestHandlerGivesTheControlPlaneTheDecodedRequest(t *testing.T) {
	// Given a control plane that keeps the request it is asked about
	var asked proxias.CompletionRequest
	handler := openai.Handler(controlPlaneFunc(func(_ context.Context, req proxias.CompletionRequest) (proxias.CompletionResponse, error) {
		asked = req
		return proxias.CompletionResponse{}, nil
	}))

	// When a request is posted
	postCompletionRequest(t, handler, strings.NewReader(completionRequestBody))

	// Then it is the decoded one, with the body kept byte for byte
	want, err := openai.DecodeRequest(strings.NewReader(completionRequestBody))
	require.NoError(t, err)
	assert.Equal(t, want, asked)
}

func TestHandlerClosesTheBodyAndFlushesAsItReads(t *testing.T) {
	// Given a streamed response, in three chunks
	stream := &spyBody{Reader: &chunks{"data: a\n\n", "data: b\n\n", "data: [DONE]\n\n"}}
	handler := openai.Handler(controlPlaneAnswering(proxias.CompletionResponse{
		Status:      http.StatusOK,
		ContentType: "text/event-stream",
		Body:        stream,
	}))

	// When a request is posted
	rec := postCompletionRequest(t, handler, strings.NewReader(completionRequestBody))

	// Then the stream is written whole, flushed, and its body is closed
	assert.Equal(t, "data: a\n\ndata: b\n\ndata: [DONE]\n\n", rec.Body.String())
	assert.True(t, rec.Flushed, "the response was flushed")
	assert.True(t, stream.closed, "the body was closed")
}
