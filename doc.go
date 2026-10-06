// Package proxias intercepts what a message asks for before an LLM sees it: an IntentClassifier names the
// intent, the Interceptor runs the compiled Workflows it asks for, and only a message with no intent (or a
// failing classifier) goes to the Fallback, usually an LLM. It knows nothing of any host application:
// whatever the host needs to pass along travels in Request.Meta.
package proxias
