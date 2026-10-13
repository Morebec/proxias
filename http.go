package proxias

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

// maxRequestBytes is the largest request body that Handler reads.
const maxRequestBytes = 32 << 20

// This file is the HTTP side of the model of Proxias: reading a chat completions request received over
// HTTP, writing the response, and serving a ControlPlane as the endpoint of the OpenAI API. A host that
// writes its own handler uses the first two.

// DecodeChatCompletionRequest reads a chat completions request received over HTTP: its body, and its headers, of
// which it keeps a copy. An error is returned if the body cannot be read,
// is not a JSON chat completions request, has no messages, or has a message whose content is neither a
// string, a list nor null.
//
// A host limits the size of the body (http.MaxBytesReader), and finds the error of a body that is too large
// with errors.As.
func DecodeChatCompletionRequest(r *http.Request) (ChatCompletionRequest, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return ChatCompletionRequest{}, fmt.Errorf("proxias: read the request: %w", err)
	}

	var req ChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return ChatCompletionRequest{}, fmt.Errorf("proxias: decode the request: %w", err)
	}
	if len(req.Messages) == 0 {
		return ChatCompletionRequest{}, errors.New("proxias: the request has no messages")
	}

	req.Header = r.Header.Clone()
	return req, nil
}

// WriteChatCompletionResponse writes the response to w: its headers, its status (200 if it has none, 500 if it is not one) and
// its body, flushed as it is read so that a stream reaches the client as it is produced. It closes the Body. A
// failure to write, the client having gone away, or to read the Body after the status is sent, has nothing
// left to tell: the response ends there.
func WriteChatCompletionResponse(w http.ResponseWriter, r ChatCompletionResponse) {
	if r.Body != nil {
		defer func() { _ = r.Body.Close() }()
	}

	for name, values := range r.Header {
		w.Header()[name] = values
	}
	status := r.Status
	switch {
	case status == 0:
		status = http.StatusOK
	case status < 100 || status > 999:
		status = http.StatusInternalServerError
	}
	w.WriteHeader(status)

	if r.Body == nil {
		return
	}

	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, err := r.Body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

// NewChatCompletionHandler makes an http.Handler that serves controlPlane over HTTP, as the chat completions endpoint of the OpenAI API: mount it on
// "POST /v1/chat/completions".
//
// It reads the request with DecodeChatCompletionRequest (400 if it cannot, 413 if it is too large), has the control plane
// process it, and writes the ChatCompletionResponse back as is with WriteChatCompletionResponse: its status, its headers
// and its body, which is flushed as it is read, so that a stream reaches the client as it is produced. The error of a control plane that fails is not sent to the client (502): it is the host's to
// log, for instance with a decorator of the control plane.
func NewChatCompletionHandler(controlPlane ControlPlane) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
		req, err := DecodeChatCompletionRequest(r)
		if err != nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				http.Error(w, "the request is too large", http.StatusRequestEntityTooLarge)
				return
			}
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		response, err := controlPlane.Process(r.Context(), req)
		if err != nil {
			http.Error(w, "the request could not be answered", http.StatusBadGateway)
			return
		}

		WriteChatCompletionResponse(w, response)
	})
}
