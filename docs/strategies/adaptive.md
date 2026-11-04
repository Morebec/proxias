# Adaptive

The adaptive strategy chooses, for each task, the strategy that suits it among named candidates, and the task is that
strategy's from start to end. A greeting can go to ReAct, a lookup chained to an email to ReWOO, a longer task that
must adapt to Observe-Plan-Execute.

## How a turn runs

1. **Choose.** When the turn follows an outside event (a person's message, a probe, an event of another agent:
   anything but the agent's own events, the results of its actions and the system's messages), the `StrategySelector`
   chooses a candidate, by name, from the session and the candidates' descriptions. The choice is yielded as a thought
   of the agent:

   ```json
   {"strategy":"rewoo"}
   ```

   When the selector chooses none, or a name no candidate has, the task is the default candidate's: `Default`, or
   the first candidate when it is empty. The recorded choice names the strategy that runs.
2. **Delegate.** The turn is the chosen strategy's `Invoke`, given the session as it was (the choice is the agent's,
   not the task's).

The turns that follow the agent's own events (its plan, a step's result, under `Agent.Loop`) find the choice the
session records and keep to it, without asking the selector, until the next outside event begins a new task.

## Selectors

- `LLMStrategySelector`: one model call, given `DefaultLLMStrategySelectionPrompt` (or its `Prompt`), who the agent is,
  the candidates (name and description) and the agent's actions (without their schemas), and the session. It answers
  `{"strategy":"<name>"}`, or `{"strategy":""}` when none clearly fits.
- `System1StrategySelector`: a `System1Model` choice among the candidates, by their descriptions, over the session as
  JSON. It keeps the choice when its confidence reaches `SelectionThreshold` (`DefaultSystem1SelectionThreshold`, 0.5,
  when nil; a pointer to 0 keeps any choice), and chooses none under it. A fast, local System One model makes the
  choice cost next to nothing.
- Any `StrategySelector` of the app's: rules, a keyword match, a fine-tuned classifier.

## Events

- `cognition.thought`: the choice, `{"strategy":"<name>"}`, once a task.
- Then the chosen strategy's events (see [ReAct](react.md), [ReWOO](rewoo.md),
  [Observe-Plan-Execute](observe-plan-execute.md)).

## Usage

```go
llm := anthropic.NewLLM(anthropicsdk.NewClient(), "claude-haiku-5-5")
strategy := proxias.NewAdaptiveStrategy(
	proxias.LLMStrategySelector{LLM: llm},
	proxias.StrategyCandidate{
		Name:        "react",
		Description: "Decides each step after seeing the last one's result: conversations, open-ended tasks.",
		Strategy:    proxias.NewReActStrategy(llm),
	},
	proxias.StrategyCandidate{
		Name:        "rewoo",
		Description: "Plans every step up front, then answers: tasks whose steps are known from the start.",
		Strategy:    proxias.NewReWOOStrategy(llm),
	},
	proxias.StrategyCandidate{
		Name:        "observe-plan-execute",
		Description: "Revises a written plan after every step: longer tasks that must adapt.",
		Strategy:    proxias.NewObservePlanExecuteStrategy(llm),
	},
)
stream, err := agent.Loop(ctx, session, strategy)
```

The first candidate is the default; set `Default` to name another.

## Trade-offs

- The selector costs one choice a task, on top of the chosen strategy's calls: one small model call with
  `LLMStrategySelector`, next to nothing with a fast `System1StrategySelector`.
- Each task gets the strategy that fits it, so a mix of tasks costs less than running all of them on the most adaptive
  strategy.
- A wrong choice holds for the whole task: the strategy is not reconsidered until the next outside event.
- The candidates' descriptions are what the selector chooses by: write them as when each one suits.
