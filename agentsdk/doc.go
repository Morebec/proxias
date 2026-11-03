// Package agentsdk is an SDK of agents that live through events rather than
// chat messages.
//
// An Event is an entry of an agent's experience: a message from a person,
// another agent or a system (an instruction is a message from a system source),
// a thought, an action it requests, the result of one, anything the app appends
// for it. A Session is the trace of those events, kept by the app, like the
// conversation of a chat completions API. An Agent is who it is, its execution
// layer (ActionExecutor) and the AgentMiddleware that wraps every turn it
// takes.
//
// How an agent thinks is a CognitiveStrategy, one turn at a time:
// NewReActStrategy (one model call, every action a tool), NewGatedStrategy (an
// ActionClassifier, an LLMActionClassifier in one call or a
// CompositeActionClassifier of an ActionSelector and an
// ActionParameterExtractor, finds the executions, the agent runs them, one
// synthesis), or the app's own. Each prompt is a field of what uses it, its
// default an exported constant (DefaultReActPrompt...). Strategies are
// vendor-neutral over two model interfaces: LLM, and System1Model (a System One
// model: questions about
// a state, answered with probabilities), which adapters implement.
//
// Agent.Invoke takes one turn, Agent.Loop takes turns until one completes,
// executing the actions requested between them, and Agent.Watch Loops on each
// event of a channel. Each returns a SessionStream (whole events, and chunks of
// text as it is written); SessionStream.Collect reads it to its end. How agents
// talk to each other is the app's: an
// action of its own that invokes another agent.
package agentsdk
