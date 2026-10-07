package openai

import (
	"errors"
	"net/http"

	"github.com/morebec/proxias"
)

// maxRequestBytes is the largest request body that Handler reads.
const maxRequestBytes = 32 << 20

// Handler serves controlPlane over HTTP, as the chat completions endpoint of the OpenAI API: mount it on
// "POST /v1/chat/completions".
//
// It decodes the request with DecodeRequest (400 if it cannot, 413 if it is too large), has the control plane
// process it, and writes the CompletionResponse back as is: its status, its content type and its body, which
// is flushed as it is read, so that a stream reaches the client as it is produced. The error of a
// control plane that fails is not sent to the client (502): it is the host's to log, for instance with a
// decorator of the control plane.
func Handler(controlPlane proxias.ControlPlane) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		req, err := DecodeRequest(http.MaxBytesReader(w, r.Body, maxRequestBytes))
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

		writeResponse(w, response)
	})
}

func writeResponse(w http.ResponseWriter, response proxias.CompletionResponse) {
	if response.Body != nil {
		defer func() { _ = response.Body.Close() }()
	}

	if response.ContentType != "" {
		w.Header().Set("Content-Type", response.ContentType)
	}
	status := response.Status
	switch {
	case status == 0:
		status = http.StatusOK
	case status < 100 || status > 999:
		status = http.StatusInternalServerError
	}
	w.WriteHeader(status)

	if response.Body == nil {
		return
	}

	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, err := response.Body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			// io.EOF, or a failure mid-stream: the status is sent already, there is nothing left to tell.
			return
		}
	}
}
