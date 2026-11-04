# ReWOO

ReWOO (Reasoning WithOut Observation) plans the whole task up front: the model writes the plan, the agent runs it
without asking the model again, and the model then writes the answer from what the plan found.

## How a turn runs

1. **Plan.** One model call, given the plan prompt, who the agent is, the inventory of the agent's actions (as JSON,
   with their schemas) and the session. It answers the plan as JSON:

   ```json
   {"steps":[
     {"id":"E1","action":"find_customer","input":{"name":"Globex"}},
     {"id":"E2","action":"send_email","input":{"to":"#E1.email","body":"Hello"}}
   ]}
   ```

   A string of an input can name what an earlier step found: `#E1.email` is replaced by the field `email` of E1's
   result data (a string as it is, any other value as JSON), and `#E1` by its whole data, as JSON. A reference to a
   step that did not run stays as it is. An agent without actions is asked no plan.
2. **Work.** The agent runs the steps in order, each its own `ActionExecutor.Execute` call, its references replaced.
   A step whose result's status is `failed` stops the work: the steps after it are not run.
3. **Solve.** One model call, given the solve prompt, who the agent is, the session, and the plan with what each step
   found (its status and data, or "not run"). Its text is streamed, and the turn ends with the model's stop reason.

## Events

- `cognition.thought`: the plan, as its JSON, when it has steps.
- `action.request`: one per step run, its `CausationID` the plan's.
- `action.result`: one per request, from `system:actions`, its `CausationID` the request's.
- `cognition.thought`: the model's thinking during the answer, kept whole.
- `message`: the answer.

## Usage

```go
llm := anthropic.NewLLM(anthropicsdk.NewClient(), "claude-haiku-5-5")
strategy := proxias.NewReWOOStrategy(llm)
stream, err := agent.Loop(ctx, session, strategy)
```

`PlanPrompt` replaces `DefaultReWOOPlanPrompt`, and `SolvePrompt` replaces `DefaultReWOOSolvePrompt`.

## Trade-offs

- Two model calls a turn, however many steps: the model reads the conversation twice, not once per step.
- The plan never sees a result. When a step returns something unexpected, the plan does not change; a failed step
  stops the work, and the answer says what was not done.
- Steps run one after another, each its own execution.
