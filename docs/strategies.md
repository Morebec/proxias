# Strategies

A cognitive strategy is how an agent thinks during a turn: what it asks its model, what it runs, and what it says.
Every strategy implements `CognitiveStrategy`:

```go
type CognitiveStrategy interface {
	Invoke(ctx context.Context, inv Invocation) iter.Seq[StreamUpdate]
}
```

`Agent.Invoke` takes one turn with a strategy, and `Agent.Loop` takes turns until the strategy stops (its last update
has a `StopReason`) or the agent reaches `DefaultMaxTurns`. A turn yields events (thoughts, action requests, their
results, messages) and text chunks as they come; the session keeps the events, so the next turn reasons on them. A
strategy passes the model's stop reason on as it is: it never decides one.

Every strategy is told who the agent is and what it remembers through `NewAgentPrompt`, and reads the session through
`NewLLMRequestFromSession`, so that any strategy sees the same conversation.

## Which one

| Strategy | Model calls for a task of N steps | Adapts to results | Suits |
|---|---|---|---|
| [ReAct](strategies/react.md) | about N + 1, one a turn | after every step | open-ended tasks, conversations, tasks whose next step depends on the last result |
| [ReWOO](strategies/rewoo.md) | 2, whatever N | never within a turn | tasks whose steps are known up front, chained through their results |
| [Observe-Plan-Execute](strategies/observe-plan-execute.md) | about N + 2 | after every step, against a written plan | longer tasks that must adapt while keeping to a plan |
| [Adaptive](strategies/adaptive.md) | the chosen strategy's, plus one choice a task | as the chosen strategy, the strategy fixed for the task | agents that meet tasks of different shapes |

ReAct is the default choice: it decides each step after seeing the last one's result. ReWOO trades that for fewer
calls and fewer tokens: the model plans once, the agent runs the plan, and the model answers once.
Observe-Plan-Execute sits between them: it writes a plan like ReWOO, and revises it after every step like ReAct
adapts, at the cost of a planning call per step.

Adaptive chooses one of them for each task, with a `StrategySelector` (an LLM in one call, or a System One model),
and keeps to it until the next outside event.
