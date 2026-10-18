// Package proxias is a deterministic execution plane for LLM applications. Seen as a UI, human prose is turned
// into commands or queries before an LLM sees it, and their results are turned back into prose.
//
// The ControlPlane is the entry point. An action is whatever an ActionExecutionEngine runs for a request: a
// plain function as much as a workflow, a process or a pipeline. An IntentClassifier, ideally a fast System-1
// component, names which of the engine's actions a request asks for and with which parameters (a
// CompositeIntentClassifier splits it into a selector and a parameter extractor). The
// ActionExecutionEngine runs them, and an ActionExecutionSynthesizer turns their results back into a
// response, usually in a single LLM pass. A request that asks for no action is answered by the ReasoningEngine,
// usually an LLM, as is.
//
// Proxias is a library: transports (gRPC, CLI...) and every application concern live in the host. The
// model of Proxias is the OpenAI chat completions API, which NewChatCompletionHandler serves over net/http, and
// the anthropic package implements the components with Claude.
package proxias
