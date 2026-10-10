package proxias_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

// failingReader is a body that cannot be read, as a connection that drops mid-request is.
type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("connection reset") }

func TestDecodeChatCompletionRequest(t *testing.T) {
	type expect struct {
		request proxias.ChatCompletionRequest // without RawRequest, which is the body in every case, and Header
		err     assert.ErrorAssertionFunc
	}
	falseValue := false
	text := func(s string) proxias.ChatContent {
		return proxias.ChatContent{{Type: proxias.ChatContentPartTypeText, Text: s}}
	}
	tests := []struct {
		name  string
		given string // the body
		then  expect
		// NOTE All tests ensure that the RawRequest field is correctly populated with the original body.
	}{
		{
			name: "canonical",
			given: `{
				"model":"m",
				"stream":true,
				"stream_options":{"include_usage":true},
				"max_completion_tokens":100,
				"stop":["END"],
				"tool_choice":"auto",
				"parallel_tool_calls":false,
				"tools":[{"type":"function","function":{"name":"restart","description":"d","parameters":{"type":"object"}}}],
				"messages":[
					{"role":"user","content":"restart api"},
					{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"restart","arguments":"{\"service\":\"api\"}"}}]},
					{"role":"tool","tool_call_id":"c1","content":"done"}
				]}`,
			then: expect{
				request: proxias.ChatCompletionRequest{
					Model:               "m",
					Stream:              true,
					StreamOptions:       proxias.ChatStreamOptions{IncludeUsage: true},
					MaxCompletionTokens: 100,
					Stop:                json.RawMessage(`["END"]`),
					ToolChoice:          json.RawMessage(`"auto"`),
					ParallelToolCalls:   &falseValue,
					Tools: []proxias.ChatTool{{
						Type:     "function",
						Function: proxias.ChatFunction{Name: "restart", Description: "d", Parameters: map[string]any{"type": "object"}},
					}},
					Messages: []proxias.ChatMessage{
						{Role: proxias.ChatMessageRoleUser, Content: text("restart api")},
						{Role: proxias.ChatMessageRoleAssistant, ToolCalls: []proxias.ChatToolCall{{
							ID:       "c1",
							Type:     "function",
							Function: proxias.ChatFunctionCall{Name: "restart", Arguments: `{"service":"api"}`},
						}}},
						{Role: proxias.ChatMessageRoleTool, ToolCallID: "c1", Content: text("done")},
					},
				},
				err: assert.NoError,
			},
		},
		{
			name:  "GIVEN a field that the request does not name THEN it is kept in the body, byte for byte",
			given: `{"model":"m","unknown_field":1,"messages":[{"role":"user","content":"hello"}]}`,
			then: expect{
				request: proxias.ChatCompletionRequest{
					Model:    "m",
					Messages: []proxias.ChatMessage{{Role: proxias.ChatMessageRoleUser, Content: text("hello")}},
				},
				err: assert.NoError,
			},
		},
		{
			name:  "GIVEN a message without content, or with a role that is not known, THEN they are kept as they are",
			given: `{"messages":[{"role":"user"},{"role":"critic","content":"hm"}]}`,
			then: expect{
				request: proxias.ChatCompletionRequest{Messages: []proxias.ChatMessage{
					{Role: proxias.ChatMessageRoleUser},
					{Role: "critic", Content: text("hm")},
				}},
				err: assert.NoError,
			},
		},
		{
			name:  "GIVEN a body that is not JSON THEN it fails",
			given: `hello`,
			then:  expect{err: assert.Error},
		},
		{
			name:  "GIVEN a body that is not an object THEN it fails",
			given: `[]`,
			then:  expect{err: assert.Error},
		},
		{
			name:  "GIVEN no messages THEN it fails",
			given: `{"model":"m"}`,
			then:  expect{err: assert.Error},
		},
		{
			name:  "GIVEN an empty list of messages THEN it fails",
			given: `{"messages":[]}`,
			then:  expect{err: assert.Error},
		},
		{
			name:  "GIVEN a content that is neither a string, a list nor null THEN it fails",
			given: `{"messages":[{"role":"user","content":12}]}`,
			then:  expect{err: assert.Error},
		},
		{
			name:  "GIVEN content parts that are not objects THEN it fails",
			given: `{"messages":[{"role":"user","content":["text"]}]}`,
			then:  expect{err: assert.Error},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/v1/chat/completions", strings.NewReader(tt.given))

			got, err := proxias.DecodeChatCompletionRequest(r)

			tt.then.err(t, err)
			if err != nil {
				assert.Zero(t, got)
				return
			}
			want := tt.then.request
			want.RawRequest = []byte(tt.given)
			want.Header = http.Header{}
			assert.Equal(t, want, got)
		})
	}
}

func TestDecodeChatCompletionRequestKeepsAHeadersCopy(t *testing.T) {
	const body = `{"model":"m","messages":[{"role":"user","content":"hello"}]}`
	type args struct {
		body   io.Reader
		header http.Header
	}
	type expect struct {
		header http.Header
		err    string
	}
	tests := []struct {
		name string
		when args
		then expect
	}{
		{
			name: "canonical",
			when: args{
				body:   strings.NewReader(body),
				header: http.Header{"Authorization": {"Bearer secret"}, "X-Request-Id": {"r1"}},
			},
			then: expect{header: http.Header{"Authorization": {"Bearer secret"}, "X-Request-Id": {"r1"}}},
		},
		{
			name: "GIVEN a body that is not a request THEN it fails",
			when: args{body: strings.NewReader(`hello`)},
			then: expect{err: "proxias: decode the request"},
		},
		{
			name: "GIVEN a body that cannot be read THEN it fails with the reader's error",
			when: args{body: failingReader{}},
			then: expect{err: "proxias: read the request: connection reset"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/v1/chat/completions", tt.when.body)
			for name, values := range tt.when.header {
				r.Header[name] = values
			}

			got, err := proxias.DecodeChatCompletionRequest(r)

			if tt.then.err != "" {
				assert.ErrorContains(t, err, tt.then.err)
				assert.Zero(t, got)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, "m", got.Model)
			assert.Equal(t, body, string(got.RawRequest), "the body, byte for byte")
			assert.Equal(t, tt.then.header, got.Header)

			r.Header.Set("X-Request-Id", "changed")
			assert.Equal(t, "r1", got.Header.Get("X-Request-Id"), "the headers are a copy")
		})
	}
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

func TestWriteChatCompletionResponse(t *testing.T) {
	type expect struct {
		status      int
		contentType string
		body        string
	}
	tests := []struct {
		name  string
		given proxias.ChatCompletionResponse
		then  expect
	}{
		{
			name: "canonical",
			given: proxias.ChatCompletionResponse{
				Status: http.StatusCreated,
				Header: http.Header{"Content-Type": {"application/json"}},
				Body:   io.NopCloser(strings.NewReader(`{"ok":true}`)),
			},
			then: expect{
				status:      http.StatusCreated,
				contentType: "application/json",
				body:        `{"ok":true}`,
			},
		},
		{
			name: "GIVEN a response without a status THEN it is written as a 200",
			given: proxias.ChatCompletionResponse{
				Header: http.Header{"Content-Type": {"text/plain"}},
				Body:   io.NopCloser(strings.NewReader("hi")),
			},
			then: expect{
				status:      http.StatusOK,
				contentType: "text/plain",
				body:        "hi",
			},
		},
		{
			name: "GIVEN a response with a status that is not one THEN it is written as a 500",
			given: proxias.ChatCompletionResponse{
				Status: 1000,
				Body:   io.NopCloser(strings.NewReader("hi")),
			},
			then: expect{
				status: http.StatusInternalServerError,
				body:   "hi",
			},
		},
		{
			name:  "GIVEN a response without a body THEN only its status is written",
			given: proxias.ChatCompletionResponse{Status: http.StatusNoContent},
			then:  expect{status: http.StatusNoContent},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := httptest.NewRecorder()

			proxias.WriteChatCompletionResponse(rec, tt.given)

			assert.Equal(t, tt.then.status, rec.Code)
			assert.Equal(t, tt.then.contentType, rec.Header().Get("Content-Type"))
			assert.Equal(t, tt.then.body, rec.Body.String())
		})
	}
}

func TestWriteChatCompletionResponseClosesTheBodyAndFlushesAsItReads(t *testing.T) {
	// Given a streamed response, in three chunks
	stream := &spyBody{Reader: &chunks{"data: a\n\n", "data: b\n\n", "data: [DONE]\n\n"}}
	response := proxias.ChatCompletionResponse{
		Status: http.StatusOK,
		Header: http.Header{"Content-Type": {"text/event-stream"}},
		Body:   stream,
	}
	rec := httptest.NewRecorder()

	// When it is written
	proxias.WriteChatCompletionResponse(rec, response)

	// Then the stream is written whole, flushed, and its body is closed
	assert.Equal(t, "data: a\n\ndata: b\n\ndata: [DONE]\n\n", rec.Body.String())
	assert.True(t, rec.Flushed, "the response was flushed")
	assert.True(t, stream.closed, "the body was closed")
}

type controlPlaneFunc func(context.Context, proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error)

func (f controlPlaneFunc) Process(ctx context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
	return f(ctx, req)
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
var unaskedControlPlane = controlPlaneFunc(func(context.Context, proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
	panic("the control plane must not be asked about a request the handler does not accept")
})

// controlPlaneAnswering is a control plane that answers response to every request.
func controlPlaneAnswering(response proxias.ChatCompletionResponse) controlPlaneFunc {
	return func(context.Context, proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
		return response, nil
	}
}

// postChatCompletionRequest posts body to the completions endpoint.
func postChatCompletionRequest(t *testing.T, handler http.Handler, body io.Reader) *httptest.ResponseRecorder {
	t.Helper()

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/v1/chat/completions", body))

	return rec
}

func TestChatCompletionHandlerRejectsWhatItCannotProcess(t *testing.T) {
	// The control plane panics if it is asked: none of these requests reaches it.
	handler := proxias.NewChatCompletionHandler(unaskedControlPlane)
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
				body:        "proxias: decode the request: invalid character 'r' looking for beginning of value\n",
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

func TestChatCompletionHandlerAnswersABadGatewayWhenTheControlPlaneFails(t *testing.T) {
	// Given a control plane that fails, naming an internal address
	handler := proxias.NewChatCompletionHandler(controlPlaneFunc(func(context.Context, proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
		return proxias.ChatCompletionResponse{}, errors.New("the classifier at 10.0.0.7 is down")
	}))

	// When a request is posted
	rec := postChatCompletionRequest(t, handler, strings.NewReader(completionRequestBody))

	// Then it is a 502 that does not say why
	assert.Equal(t, http.StatusBadGateway, rec.Code)
	assert.Equal(t, "text/plain; charset=utf-8", rec.Header().Get("Content-Type"))
	assert.Equal(t, "the request could not be answered\n", rec.Body.String())
}

func TestChatCompletionHandlerGivesTheControlPlaneTheDecodedRequest(t *testing.T) {
	// Given a control plane that keeps the request it is asked about
	var asked proxias.ChatCompletionRequest
	handler := proxias.NewChatCompletionHandler(controlPlaneFunc(func(_ context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
		asked = req
		return proxias.ChatCompletionResponse{}, nil
	}))

	// When a request is posted, with a header
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/v1/chat/completions", strings.NewReader(completionRequestBody))
	req.Header.Set("X-Request-Id", "r1")
	handler.ServeHTTP(httptest.NewRecorder(), req)

	// Then it is the decoded one, with the body kept byte for byte and the header
	assert.Equal(t, completionRequestBody, string(asked.RawRequest))
	assert.Equal(t, "m", asked.Model)
	assert.Equal(t, "r1", asked.Header.Get("X-Request-Id"))
}

func TestChatCompletionHandlerWritesTheControlPlanesResponse(t *testing.T) {
	// Given a control plane that answers a stream
	handler := proxias.NewChatCompletionHandler(controlPlaneAnswering(proxias.ChatCompletionResponse{
		Status: http.StatusCreated,
		Header: http.Header{"Content-Type": {"text/event-stream"}},
		Body:   io.NopCloser(strings.NewReader("data: [DONE]\n\n")),
	}))

	// When a request is posted
	rec := postChatCompletionRequest(t, handler, strings.NewReader(completionRequestBody))

	// Then it is written as it is: the details of how are those of ChatCompletionResponse.WriteHTTP
	assert.Equal(t, http.StatusCreated, rec.Code)
	assert.Equal(t, "text/event-stream", rec.Header().Get("Content-Type"))
	assert.Equal(t, "data: [DONE]\n\n", rec.Body.String())
}
