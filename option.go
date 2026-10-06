package proxias

import (
	"context"
	"log/slog"
)

// Hooks observe what the Interceptor does, for tracing and metrics. Any may be nil.
type Hooks struct {
	// Classified is called with the Intent (NoIntent included) and the classifier's error, if any, and how
	// long it took in nanoseconds.
	Classified func(ctx context.Context, request Request, intent Intent, err error, nanos int64)
	// Ran is called after each Workflow, with its Result and the error it returned, if any.
	Ran func(ctx context.Context, result Result, err error, nanos int64)
}

// Option configures an Interceptor.
type Option func(*Interceptor)

// WithIntentClassifier sets the IntentClassifier. Without one, nothing is intercepted.
func WithIntentClassifier(classifier IntentClassifier) Option {
	return func(i *Interceptor) { i.classifier = classifier }
}

// WithFallback sets the Fallback.
func WithFallback(fallback Fallback) Option { return func(i *Interceptor) { i.fallback = fallback } }

// WithHooks sets the Hooks.
func WithHooks(hooks Hooks) Option { return func(i *Interceptor) { i.hooks = hooks } }

// WithLogger sets the logger, slog.Default() otherwise. Nothing of a message is ever logged.
func WithLogger(logger *slog.Logger) Option { return func(i *Interceptor) { i.logger = logger } }

// WithInput sets how the input of the Workflows of an Intent is made from the Request, when the Intent
// has none. The default is {"message": "<last message>"}.
func WithInput(input func(Request) []byte) Option { return func(i *Interceptor) { i.input = input } }
