package proxias

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"regexp"
	"slices"
	"strings"

	"github.com/google/uuid"
)

// ReWOOCognitiveStrategy is the ReWOO strategy (Reasoning WithOut
// Observation): the model plans the whole task up front, the agent runs the
// plan without asking the model again, and the model then writes the answer
// from what the plan found.
//
// A turn has three phases:
//   - plan: one model call, given the agent's actions, writes the plan as
//     steps, each an action with its input; an input can name the result of
//     an earlier step ("#E1"). The plan is yielded as a thought of the agent.
//   - work: the agent runs the steps in order, each step's references
//     replaced by the results they name, and yields each request and its
//     result. A failed step stops the work.
//   - solve: one model call writes the reply from the session, the plan and
//     what each step found. It is streamed, and ends the turn.
//
// So a turn is two model calls, however many steps the plan has, and the model
// reads the conversation twice instead of once per step. It suits tasks whose
// steps are known up front, such as a lookup and then an email built from it.
// Its trade-off is that the plan never sees a result: when a step returns
// something unexpected, the plan does not change; ReAct, which decides each
// step after the last one's result, adapts where ReWOO cannot.
type ReWOOCognitiveStrategy struct {
	LLM LLM
	// PlanPrompt is the system prompt of the plan: DefaultReWOOPlanPrompt when
	// empty.
	PlanPrompt string
	// SolvePrompt is the system prompt of the answer: DefaultReWOOSolvePrompt
	// when empty.
	SolvePrompt string
}

// NewReWOOStrategy returns the ReWOO strategy of the model.
func NewReWOOStrategy(llm LLM) ReWOOCognitiveStrategy {
	return ReWOOCognitiveStrategy{LLM: llm}
}

// reWOOPlan is a plan of the ReWOO strategy, as the model writes it.
type reWOOPlan struct {
	Steps []reWOOStep `json:"steps"`
}

// reWOOStep is a step of a ReWOO plan: an action and its input, whose string
// values can name the result of an earlier step ("#E1").
type reWOOStep struct {
	ID     string         `json:"id"`
	Action string         `json:"action"`
	Input  map[string]any `json:"input,omitempty"`
}

// Invoke takes one ReWOO turn: the model plans, the agent works through the
// plan, and the model solves the task from what the plan found.
func (s ReWOOCognitiveStrategy) Invoke(ctx context.Context, inv Invocation) iter.Seq[StreamUpdate] {
	return func(yield func(StreamUpdate) bool) {
		plan, err := s.plan(ctx, inv)
		if err != nil {
			yield(StreamUpdate{Err: err})
			return
		}

		evidence, ok := s.work(ctx, inv, plan, yield)
		if !ok {
			return
		}

		s.solve(ctx, inv, plan, evidence, yield)
	}
}

// plan has the model write the plan, given the agent's actions. An agent
// without actions has an empty plan, asked of no model.
func (s ReWOOCognitiveStrategy) plan(ctx context.Context, inv Invocation) (reWOOPlan, error) {
	var plan reWOOPlan
	if inv.Agent.ActionExecutor == nil {
		return plan, nil
	}

	descriptors, err := inv.Agent.ActionExecutor.Descriptors(ctx, inv.Session)
	if err != nil {
		return plan, err
	}
	if len(descriptors) == 0 {
		return plan, nil
	}

	planPrompt := s.PlanPrompt
	if planPrompt == "" {
		planPrompt = DefaultReWOOPlanPrompt
	}
	agentPrompt, err := NewAgentPrompt(ctx, inv)
	if err != nil {
		return plan, err
	}
	inventory, err := json.Marshal(map[string]any{"actions": descriptors})
	if err != nil {
		return plan, err
	}
	systemInstructions := LLMMessage{
		Role: LLMMessageRoleSystem,
		Parts: []LLMPart{
			{Text: planPrompt},
			{Text: agentPrompt.String()},
			{Text: string(inventory)},
		},
	}
	llmRequest := NewLLMRequestFromSession(inv.Agent.AsSource(), inv.Session)
	llmRequest.Messages = slices.Insert(llmRequest.Messages, 0, systemInstructions)

	var text strings.Builder
	for chunk, err := range s.LLM.Complete(ctx, llmRequest) {
		if err != nil {
			return plan, fmt.Errorf("planning: %w", err)
		}
		text.WriteString(chunk.Text)
	}

	// The first JSON value of the answer: a code fence or words around it are
	// left out.
	answerText := text.String()
	start := strings.Index(answerText, "{")
	if start < 0 {
		return plan, errors.New("planning: the model's answer has no JSON object")
	}
	decoder := json.NewDecoder(strings.NewReader(answerText[start:]))
	if err := decoder.Decode(&plan); err != nil {
		return plan, fmt.Errorf("planning: the model's answer is not a plan: %w", err)
	}
	return plan, nil
}

// work yields the plan, as a thought of the agent, then has the agent run its
// steps in order, each its own execution, its references replaced by the
// results they name, and yields each request and its result. A failed step
// stops the work. It returns what each step found, by step ID, or false when
// the turn ended (an error, yielded, or a stopped stream).
func (s ReWOOCognitiveStrategy) work(
	ctx context.Context,
	inv Invocation,
	plan reWOOPlan,
	yield func(StreamUpdate) bool,
) (map[string]ActionExecutionResult, bool) {
	evidence := map[string]ActionExecutionResult{}
	if len(plan.Steps) == 0 {
		return evidence, true
	}
	if inv.Agent.ActionExecutor == nil {
		yield(StreamUpdate{Err: errors.New("the plan runs actions, but the agent has no actions")})
		return nil, false
	}

	// The plan is the model's own words: its JSON.
	payload, _ := json.Marshal(plan) // a plan decoded from JSON marshals
	planEvent := NewEvent(EventTypeThought, inv.Agent.AsSource(), payload)
	if !yield(StreamUpdate{Events: []Event{planEvent}}) {
		return nil, false
	}

	events := []Event{planEvent}
	for _, step := range plan.Steps {
		request := ActionExecutionRequest{
			ActionID: step.Action,
			Params:   withReWOOEvidence(step.Input, evidence),
		}
		requestEvent, err := NewActionExecutionRequestEvent(inv.Agent.AsSource(), request)
		if err != nil {
			yield(StreamUpdate{Err: err})
			return nil, false
		}
		requestEvent.CausationID = planEvent.ID

		session := inv.Session.WithEvents(append(events, requestEvent)...)
		results, err := inv.Agent.ActionExecutor.Execute(ctx, session, []ActionExecutionRequest{request})
		if err != nil {
			yield(StreamUpdate{Err: fmt.Errorf("executing actions: %w", err)})
			return nil, false
		}
		if len(results) != 1 {
			yield(StreamUpdate{Err: fmt.Errorf("%d results for 1 action", len(results))})
			return nil, false
		}

		resultEvent, err := NewActionExecutionResultEvent(NewSystemSource("actions"), results[0])
		if err != nil {
			yield(StreamUpdate{Err: err})
			return nil, false
		}
		resultEvent.CausationID = requestEvent.ID
		events = append(events, requestEvent, resultEvent)
		evidence[step.ID] = results[0]
		if !yield(StreamUpdate{Events: []Event{requestEvent, resultEvent}}) {
			return nil, false
		}

		// A failed step stops the work: the steps after it may need what it
		// did not find.
		if results[0].Status == "failed" {
			break
		}
	}
	return evidence, true
}

// reWOOReference is a reference to an earlier step, in a string of an input:
// "#E1", the data of its result, or "#E1.email", one of its fields.
var reWOOReference = regexp.MustCompile(`#(E[0-9]+)(?:\.([A-Za-z0-9_]+))?`)

// withReWOOEvidence is the input with each reference to a step in its string
// values replaced by what that step found.
func withReWOOEvidence(input map[string]any, evidence map[string]ActionExecutionResult) map[string]any {
	if input == nil {
		return nil
	}

	params := make(map[string]any, len(input))
	for name, value := range input {
		text, ok := value.(string)
		if !ok {
			params[name] = value
			continue
		}
		params[name] = reWOOReference.ReplaceAllStringFunc(text, func(reference string) string {
			return reWOOEvidenceOf(reference, evidence)
		})
	}
	return params
}

// reWOOEvidenceOf is what a reference names, as text: a field's value (a string
// as it is, any other value as JSON), or the whole data as JSON. A reference to
// a step that did not run stays as it is.
func reWOOEvidenceOf(reference string, evidence map[string]ActionExecutionResult) string {
	match := reWOOReference.FindStringSubmatch(reference)
	result, ran := evidence[match[1]]
	if !ran {
		return reference
	}

	var value any = result.Data
	if match[2] != "" {
		value = result.Data[match[2]]
	}
	text, isText := value.(string)
	if isText {
		return text
	}
	data, _ := json.Marshal(value) // data decoded from JSON marshals
	return string(data)
}

// solve has the model write the reply from the session, the plan and what each
// step found, streamed, and yields it with the model's stop reason as it is.
func (s ReWOOCognitiveStrategy) solve(
	ctx context.Context,
	inv Invocation,
	plan reWOOPlan,
	evidence map[string]ActionExecutionResult,
	yield func(StreamUpdate) bool,
) {
	currentAgent := inv.Agent.AsSource()
	solvePrompt := s.SolvePrompt
	if solvePrompt == "" {
		solvePrompt = DefaultReWOOSolvePrompt
	}
	agentPrompt, err := NewAgentPrompt(ctx, inv)
	if err != nil {
		yield(StreamUpdate{Err: err})
		return
	}

	// The plan and its evidence, step by step: what the model is told the
	// work found.
	var found strings.Builder
	found.WriteString("# The plan, and what each step found")
	for _, step := range plan.Steps {
		input, _ := json.Marshal(step.Input) // an input decoded from JSON marshals
		fmt.Fprintf(&found, "\n- #%s = %s(%s)", step.ID, step.Action, input)
		result, ran := evidence[step.ID]
		if !ran {
			found.WriteString(": not run")
			continue
		}
		data, _ := json.Marshal(result.Data) // data decoded from JSON marshals
		fmt.Fprintf(&found, ": %s %s", result.Status, data)
	}
	systemInstructions := LLMMessage{
		Role: LLMMessageRoleSystem,
		Parts: []LLMPart{
			{Text: solvePrompt},
			{Text: agentPrompt.String()},
			{Text: found.String()},
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
