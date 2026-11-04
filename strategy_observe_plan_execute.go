package proxias

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/google/uuid"
)

// ObservePlanExecuteCognitiveStrategy is the Observe-Plan-Execute strategy:
// the model keeps a written plan of the task, and revises it after every step,
// from what the step found.
//
// A turn is one phase, and Agent.Loop runs them in turn:
//   - observe and plan: one model call, given the agent's actions and what the
//     steps of the task found so far, writes the plan of the steps left. The
//     plan is yielded as a thought of the agent, kept in the session. When no
//     step is left, the turn reports instead.
//   - execute: when the agent's last event is its plan, the agent runs the
//     plan's first step, and yields its request and its result.
//   - report: one model call writes the answer from the session and what the
//     steps found. It is streamed, and ends the task.
//
// So a task of N steps takes about N + 2 model calls: a plan before each step,
// the last one empty, and the answer. It suits longer tasks that must adapt to
// what they find (a lookup that comes back empty, a step that fails) while
// keeping to a plan. Its trade-off is a planning call per step: ReWOO, which
// never re-plans, takes two calls whatever the number of steps, and ReAct,
// which keeps no plan, one call per step.
type ObservePlanExecuteCognitiveStrategy struct {
	LLM LLM
	// PlanPrompt is the system prompt of the plan:
	// DefaultObservePlanExecutePlanPrompt when empty.
	PlanPrompt string
	// ReportPrompt is the system prompt of the answer:
	// DefaultObservePlanExecuteReportPrompt when empty.
	ReportPrompt string
}

// NewObservePlanExecuteStrategy returns the Observe-Plan-Execute strategy of
// the model.
func NewObservePlanExecuteStrategy(llm LLM) ObservePlanExecuteCognitiveStrategy {
	return ObservePlanExecuteCognitiveStrategy{LLM: llm}
}

// observePlanExecutePlan is a plan of the Observe-Plan-Execute strategy, as the
// model writes it: the steps left, in order.
type observePlanExecutePlan struct {
	Steps []observePlanExecuteStep `json:"steps"`
}

// observePlanExecuteStep is a step of a plan: an action and its input.
type observePlanExecuteStep struct {
	Action string         `json:"action"`
	Input  map[string]any `json:"input,omitempty"`
}

// Invoke takes one phase of the task: the next step of the agent's plan when
// its last event is that plan, otherwise a plan of the steps left, or the
// report when none is left.
func (s ObservePlanExecuteCognitiveStrategy) Invoke(ctx context.Context, inv Invocation) iter.Seq[StreamUpdate] {
	return func(yield func(StreamUpdate) bool) {
		plan, planEvent, isPlanned := s.lastPlan(inv)
		if isPlanned {
			s.execute(ctx, inv, plan, planEvent, yield)
			return
		}

		s.observeAndPlan(ctx, inv, yield)
	}
}

// lastPlan is the agent's plan and its event, when it is the last event of
// the session.
func (s ObservePlanExecuteCognitiveStrategy) lastPlan(inv Invocation) (observePlanExecutePlan, Event, bool) {
	var plan observePlanExecutePlan
	last := inv.LastEvent()
	if last.Type != EventTypeThought {
		return plan, Event{}, false
	}
	if last.Source != inv.Agent.AsSource() {
		return plan, Event{}, false
	}

	err := json.Unmarshal(last.Payload, &plan)
	if err != nil {
		return plan, Event{}, false
	}
	return plan, last, len(plan.Steps) > 0
}

// observeAndPlan has the model write the plan of the steps left, from what the
// steps found so far, and yields it. When no step is left, or the agent has no
// actions, it reports instead.
func (s ObservePlanExecuteCognitiveStrategy) observeAndPlan(
	ctx context.Context,
	inv Invocation,
	yield func(StreamUpdate) bool,
) {
	if inv.Agent.ActionExecutor == nil {
		s.report(ctx, inv, yield)
		return
	}
	descriptors, err := inv.Agent.ActionExecutor.Descriptors(ctx, inv.Session)
	if err != nil {
		yield(StreamUpdate{Err: err})
		return
	}
	if len(descriptors) == 0 {
		s.report(ctx, inv, yield)
		return
	}

	planPrompt := s.PlanPrompt
	if planPrompt == "" {
		planPrompt = DefaultObservePlanExecutePlanPrompt
	}
	agentPrompt, err := NewAgentPrompt(ctx, inv)
	if err != nil {
		yield(StreamUpdate{Err: err})
		return
	}
	inventory, err := json.Marshal(map[string]any{"actions": descriptors})
	if err != nil {
		yield(StreamUpdate{Err: err})
		return
	}
	systemInstructions := LLMMessage{
		Role: LLMMessageRoleSystem,
		Parts: []LLMPart{
			{Text: planPrompt},
			{Text: agentPrompt.String()},
			{Text: string(inventory)},
			{Text: s.observedSteps(inv)},
		},
	}
	llmRequest := NewLLMRequestFromSession(inv.Agent.AsSource(), inv.Session)
	llmRequest.Messages = slices.Insert(llmRequest.Messages, 0, systemInstructions)

	var text strings.Builder
	for chunk, err := range s.LLM.Complete(ctx, llmRequest) {
		if err != nil {
			yield(StreamUpdate{Err: fmt.Errorf("planning: %w", err)})
			return
		}
		text.WriteString(chunk.Text)
	}

	// The first JSON value of the answer: a code fence or words around it are
	// left out.
	answerText := text.String()
	start := strings.Index(answerText, "{")
	if start < 0 {
		yield(StreamUpdate{Err: errors.New("planning: the model's answer has no JSON object")})
		return
	}
	var plan observePlanExecutePlan
	decoder := json.NewDecoder(strings.NewReader(answerText[start:]))
	if err := decoder.Decode(&plan); err != nil {
		yield(StreamUpdate{Err: fmt.Errorf("planning: the model's answer is not a plan: %w", err)})
		return
	}

	// No step left: the task is done, and the turn reports.
	if len(plan.Steps) == 0 {
		s.report(ctx, inv, yield)
		return
	}

	// The plan is the model's own words: its JSON. The next turn runs its
	// first step.
	payload, _ := json.Marshal(plan) // a plan decoded from JSON marshals
	yield(StreamUpdate{Events: []Event{NewEvent(EventTypeThought, inv.Agent.AsSource(), payload)}})
}

// execute has the agent run the plan's first step, the plan its cause, and
// yields its request and its result, for the next turn to observe.
func (s ObservePlanExecuteCognitiveStrategy) execute(
	ctx context.Context,
	inv Invocation,
	plan observePlanExecutePlan,
	planEvent Event,
	yield func(StreamUpdate) bool,
) {
	if inv.Agent.ActionExecutor == nil {
		yield(StreamUpdate{Err: errors.New("the plan runs actions, but the agent has no actions")})
		return
	}

	request := ActionExecutionRequest{
		ActionID: plan.Steps[0].Action,
		Params:   plan.Steps[0].Input,
	}
	requestEvent, err := NewActionExecutionRequestEvent(inv.Agent.AsSource(), request)
	if err != nil {
		yield(StreamUpdate{Err: err})
		return
	}
	requestEvent.CausationID = planEvent.ID

	session := inv.Session.WithEvents(requestEvent)
	results, err := inv.Agent.ActionExecutor.Execute(ctx, session, []ActionExecutionRequest{request})
	if err != nil {
		yield(StreamUpdate{Err: fmt.Errorf("executing actions: %w", err)})
		return
	}
	if len(results) != 1 {
		yield(StreamUpdate{Err: fmt.Errorf("%d results for 1 action", len(results))})
		return
	}

	resultEvent, err := NewActionExecutionResultEvent(NewSystemSource("actions"), results[0])
	if err != nil {
		yield(StreamUpdate{Err: err})
		return
	}
	resultEvent.CausationID = requestEvent.ID
	yield(StreamUpdate{Events: []Event{requestEvent, resultEvent}})
}

// report has the model write the answer from the session and what the steps
// found, streamed, and yields it with the model's stop reason as it is.
func (s ObservePlanExecuteCognitiveStrategy) report(
	ctx context.Context,
	inv Invocation,
	yield func(StreamUpdate) bool,
) {
	currentAgent := inv.Agent.AsSource()
	reportPrompt := s.ReportPrompt
	if reportPrompt == "" {
		reportPrompt = DefaultObservePlanExecuteReportPrompt
	}
	agentPrompt, err := NewAgentPrompt(ctx, inv)
	if err != nil {
		yield(StreamUpdate{Err: err})
		return
	}
	systemInstructions := LLMMessage{
		Role: LLMMessageRoleSystem,
		Parts: []LLMPart{
			{Text: reportPrompt},
			{Text: agentPrompt.String()},
			{Text: s.observedSteps(inv)},
		},
	}
	llmRequest := NewLLMRequestFromSession(currentAgent, inv.Session)
	llmRequest.Messages = slices.Insert(llmRequest.Messages, 0, systemInstructions)

	// The model's answer, its text streamed as it comes, gathered into one
	// message.
	message := LLMMessage{
		ID:   EventID(uuid.NewString()),
		Role: LLMMessageRoleAssistant,
	}
	var (
		text strings.Builder
		stop StopReason
	)
	for chunk, err := range s.LLM.Complete(ctx, llmRequest) {
		switch {
		case err != nil:
			yield(StreamUpdate{Err: err})
			return
		case chunk.StopReason != "":
			stop = chunk.StopReason
		case chunk.isThinking():
			thinking := LLMPart{
				Thinking:         chunk.Thinking,
				Signature:        chunk.Signature,
				RedactedThinking: chunk.RedactedThinking,
			}
			message.Parts = append(message.Parts, thinking)
		case chunk.Text != "":
			text.WriteString(chunk.Text)
			update := StreamUpdate{
				Chunk: &StreamChunk{
					EventID: message.ID,
					Data:    []byte(chunk.Text),
				},
			}
			if !yield(update) {
				return
			}
		}
	}
	message.Parts = append(message.Parts, LLMPart{Text: strings.TrimSpace(text.String())})

	yield(StreamUpdate{Events: NewEventsFromLLMMessage(currentAgent, message), StopReason: stop})
}

// observedSteps is what the steps of the task found so far, as the model is
// told it: each action the agent ran since the event that began the task, its
// input, and its result's status and data.
func (s ObservePlanExecuteCognitiveStrategy) observedSteps(inv Invocation) string {
	events := inv.Session.Events()
	currentAgent := inv.Agent.AsSource()

	// The task began with the last event that is neither the agent's own nor
	// the result of one of its actions.
	begin := 0
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Source == currentAgent {
			continue
		}
		if events[i].Type == EventTypeActionResult {
			continue
		}
		begin = i + 1
		break
	}

	var observed strings.Builder
	observed.WriteString("# What the steps of the task found so far")
	steps := 0
	for _, e := range events[begin:] {
		switch e.Type {
		case EventTypeActionRequest:
			fmt.Fprintf(&observed, "\n- %s", e.Text())
			steps++
		case EventTypeActionResult:
			fmt.Fprintf(&observed, ": %s", e.Text())
		}
	}
	if steps == 0 {
		observed.WriteString("\nNo step has run yet.")
	}
	return observed.String()
}
