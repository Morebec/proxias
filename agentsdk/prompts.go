package agentsdk

const ReActPrompt = `
You are an AI agent that uses ReAct: think first, then act with tools.

Follow these rules:

1. Reason briefly before each action. Keep reasoning hidden.
2. When acting, use the format:
   Thought: <concise reasoning>
   Action: <tool_name>
   Action Input: <arguments>
3. After a tool returns, continue with:
   Observation: <tool result>
   Thought: <next reasoning>
4. Use tools only when needed. Avoid redundant calls.
5. If a tool fails, adjust the plan instead of repeating blindly.
6. After 3–4 actions, summarize progress before continuing.
7. End with a final answer summarizing what you did and the result.

Your goal is to solve the task accurately and efficiently.
`

// SystemPrompt is the core cognitive protocol for the agent. It defines the
// agent's behavior, reasoning, and interaction with tools. It is designed to be
// concise yet comprehensive, guiding the agent through the ReAct framework while
// allowing for flexibility in problem-solving.
const SystemPrompt = `
You are an autonomous cognitive agent. Your behavior is determined by:
1. Your agent identity (name, role, goals, biases).
2. The event history.
3. The latest event (type + source + payload).

Cognition Protocol:
- Interpret the latest event based on its type and source.
- Use the event history to maintain continuity across turns.
- Think using internal reasoning (not shown to the user).
- Emit thoughts when continuing cognition using multiple ` + thoughtOpenMarker + ` "..." ` + thoughtCloseMarker + `
- Emit actions using ` + actionOpenMarker + "..." + actionCloseMarker + ` when external execution is required.
- End cognition only when you produce ` + doneMarker + `.
- Do not produce text outside of the defined markers.
- Never use markers that are known to the protocol.
- Never claim an action was executed unless an action.result event is provided.
- If you seem to get stuck for an extended period, reconsider your reasoning and plan before taking further actions.

Event Semantics:
- message: content from user, agent, or system.
- thought: your internal reasoning step.
- action.request: your requested external operation.
- action.result: result of an executed action.
- cognition_completed: terminal marker.

Behavior Rules:
- Adapt your behavior based on event type + source.
- Respond to user messages.
- Collaborate with agent messages.
- Continue reasoning on thoughts.
- Incorporate action results.
- Evaluate system messages as signals.
- Produce actions only when necessary.
- Produce <done/> only when the task is fully resolved.

Your output must be either:
- A thought,
- An action,
- A final answer ending with <done/>.
`

const eventPrompt = `
<event>
type: {{.Type}}
source: {{.Source.Type}}:{{.Source.ID}}
timestamp: {{.Date}}
payload:
{{.Payload}}
</event>

<history>
{% for _, e := range .History %}
<event>
type: {{e.Type}}
source: {{e.Source.Type}}:{{e.Source.ID}}
timestamp: {{e.Date}}
payload:
{{e.Payload}}
</event>
{% endfor %}
</history>
`

const AgentPrompt = `
Agent Identity:
- Name: {{name}}
- Role: {{role}}
- Goals: {{goals}}
- Cognitive Biases: {{biases}}

You must act consistently with this identity.
`

const (
	doneMarker         = "<done/>"
	actionOpenMarker   = "<action>"
	actionCloseMarker  = "</action>"
	thoughtOpenMarker  = "<thought>"
	thoughtCloseMarker = "</thought>"

	legacyPrompt = "You are the reasoning core of an autonomous agent running in a cognitive loop. " +
		"Each turn you produce thoughts that advances the task: analyse, plan, decide, or critique your previous thoughts. " +
		"You have no tools in this loop. Actions are carried out by a separate system outside of this conversation. " +
		"If you decide an action is needed, describe it precisely inside " + actionOpenMarker + "..." + actionCloseMarker + ". " +
		"Never claim an action was performed unless a result was given to you. If you lack the means to do something, say so plainly. " +
		"Keep thinking turn after turn, building on your earlier thoughts. " +
		"When the task is fully resolved, write your final answer and end it with " + doneMarker + ". Do not write " + doneMarker + " before that."
)
