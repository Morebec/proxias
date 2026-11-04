# ReAct

ReAct (Reason and Act) interleaves thinking and acting: in each turn the model reads the conversation, then either
answers or asks for actions; the agent runs them, and the next turn reasons on their results.

## How a turn runs

1. **Reason.** One model call, given the strategy's system prompt, who the agent is (`NewAgentPrompt`), the session,
   and every action of the agent as a tool with its JSON Schema. Its text is streamed as it comes.
2. **Act.** When the model called tools, the agent runs them, all in one `ActionExecutor.Execute` call, and the turn
   ends with their results. Under `Agent.Loop`, the next turn reasons on them.

When the model calls no tool, its answer is the reply, and the turn ends with the model's stop reason.

## Events

- `cognition.thought`: the model's thinking, kept whole to be given back as it came; and its text, when it comes with
  tool calls.
- `action.request`: one per tool call, of the call's ID.
- `action.result`: one per request, from `system:actions`, its `CausationID` the request's.
- `message`: the reply, when the model calls no tool.

## Usage

```go
llm := anthropic.NewLLM(anthropicsdk.NewClient(), "claude-haiku-5-5")
agent := &proxias.Agent{
	ID:             "ada",
	AgentPersona:   proxias.AgentPersona{Name: "Ada", Role: "assistant"},
	ActionExecutor: actions,
}
session := proxias.NewSession(proxias.NewMessageEvent(proxias.NewUserSource("john"), "What time is it?"))
stream, err := agent.Loop(ctx, session, proxias.NewReActStrategy(llm))
```

`SystemPrompt` replaces `DefaultReActPrompt`.

## Trade-offs

- One model call per round of results, plus the reply: a task of N dependent steps takes about N + 1 calls, and each
  call reads the whole conversation again.
- It adapts at every step: an unexpected result changes what it does next.
- Several independent actions can come in one call (parallel tool use): they run together, in one turn.
