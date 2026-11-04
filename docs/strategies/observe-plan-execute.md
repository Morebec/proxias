# Observe-Plan-Execute

Observe-Plan-Execute keeps a written plan of the task and revises it after every step: the model plans the steps left,
the agent runs the first one, the model observes what it found and plans again, until no step is left; then it
answers.

## How a turn runs

A turn is one phase; `Agent.Loop` runs them in turn, and the session carries the plan from one to the next.

1. **Observe and plan.** When the agent's last event is not its plan (a new message, a probe, a step's result), one
   model call, given the plan prompt, who the agent is, the inventory of its actions (as JSON, with their schemas),
   what the steps of the task found so far, and the session, answers the plan of the steps left:

   ```json
   {"steps":[
     {"action":"find_customer","input":{"name":"Globex"}},
     {"action":"send_email","input":{"to":"?"}}
   ]}
   ```

   The plan is yielded as a thought of the agent, and the turn ends with no stop reason: the loop goes on. Only the
   first step runs before the next plan, so the model fills the later steps' inputs once it has seen what the earlier
   ones found; there are no references between steps. When the plan has no step (the task is done, needs no action,
   or cannot go further), or the agent has no actions, the turn reports instead.
2. **Execute.** When the agent's last event is its plan, the agent runs the plan's first step, one
   `ActionExecutor.Execute` call, and the turn ends with its request and result, for the next turn to observe. Any
   other last event, a system's message included, is new input: the turn observes and plans again.
3. **Report.** One model call, given the report prompt, who the agent is, what the steps found, and the session,
   writes the answer. Its text is streamed, and the turn ends with the model's stop reason, which ends the loop.

What the steps found is every action the agent ran since the event that began the task (the last one that is neither
the agent's own nor a result), its request and its result. A middleware that adds an instruction to every turn (a
clock, say) begins a new task each time: add it to the turns that answer a new message only.

## Events

- `cognition.thought`: each plan, as its JSON.
- `action.request`: the step run, its `CausationID` the plan's.
- `action.result`: from `system:actions`, its `CausationID` the request's.
- `cognition.thought`: the model's thinking during the answer, kept whole.
- `message`: the answer.

## Usage

```go
llm := anthropic.NewLLM(anthropicsdk.NewClient(), "claude-haiku-5-5")
strategy := proxias.NewObservePlanExecuteStrategy(llm)
stream, err := agent.Loop(ctx, session, strategy) // Loop: each phase is a turn
```

`PlanPrompt` replaces `DefaultObservePlanExecutePlanPrompt`, and `ReportPrompt` replaces
`DefaultObservePlanExecuteReportPrompt`. A task of N steps takes about 2N + 1 turns of the loop (a plan and a step each,
then the plan that ends it and the report in one turn): keep it within `DefaultMaxTurns`.

## Trade-offs

- About N + 2 model calls for a task of N steps: a plan before each step, the last one empty, and the answer.
- It adapts to every result: a lookup that comes back empty or a step that fails changes the plan.
- It keeps to a plan the model wrote down, which helps longer tasks stay on course; ReAct keeps none, and ReWOO never
  revises its plan.
- A step runs alone: steps that could run together still take a turn each.
