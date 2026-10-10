<p align="center"><img src="assets/logo.svg" alt="Proxias" width="120"></p>

# Proxias

A deterministic execution plane for LLM applications, as an embeddable Go SDK.


## Why

The usual agent is a ReAct loop: the LLM reads the request, picks a tool (often over MCP), reads its output, picks another, and so on until it answers. For the many requests that code can serve directly, that loop is:

- **slow and costly**: several LLM round trips, each re-reading a context that grows with every tool output;
- **non-deterministic**: the same request may take a different path, or a wrong one, from one run to the next;
- **exposed**: every tool output is read by the model that decides which tool to call next, a wide surface for prompt injection.

Proxias flips it:

- **Deterministic execution plane.** Recognizing an intent is a **System-1** task, given to a classifier that answers in milliseconds: local rules, or a small discriminative model. If the intent matches a registered workflow, Proxias executes compiled, tested Go directly: no tool loop, no model choosing what runs.
- **Only what each step needs.** With a composite classifier, the parameter extractor sees only the descriptors of the workflows that were selected, nothing of the others. Fewer tokens, a narrower prompt-injection surface, and each step can be a small, cheap model.
- **Single-pass synthesis.** The LLM is kept for **System-2**, once, at the end: it translates the raw execution output into human language or a structured UI spec.
- **Fallback pass-through.** If no workflow matches, the request goes to the `ReasoningEngine` for fuzzy reasoning, as it would without Proxias.

Proxias is strictly an SDK: transports (HTTP `/v1/chat/completions`, gRPC, CLI) live in the host.

## Install

```sh
go get github.com/morebec/proxias
```

```go
import "github.com/morebec/proxias"
```

## The four interfaces

| Interface | It | You implement it with |
|---|---|---|
| `IntentClassifier` | **which workflows**, if any: turns a request into an `Intent`, from the workflows' descriptors | rules, a local model, a classification service |
| `WorkflowExecutionEngine` | describes the workflows (`WorkflowDescriptors`) and runs the ones an `Intent` asks for (`ExecuteWorkflows`) | your workflow registry, a job queue, a workflow service |
| `WorkflowExecutionSynthesizer` | **the reply** to a request that ran workflows: makes the response from the intent and the results | usually one LLM call, or templates |
| `ReasoningEngine` | **the reply** to a request no workflow fits: open-ended reasoning | the LLM, as it would be without Proxias |

`proxias.NewControlPlane(classifier, workflows, synthesizer, reasoner, options...)` wires them into a `ControlPlane`, whose one method is `Process`. It returns an error, never panics, if any of the four is nil. `IntentClassifierFunc` and `WorkflowExecutionSynthesizerFunc` make one of a function.

### Two stages in one classifier

Naming the workflows is one thing, reading their parameters another. `NewCompositeIntentClassifier(selector, extractor)` makes a `CompositeIntentClassifier` out of two smaller interfaces, so each can be a small, cheap model of its own:

| Stage | It |
|---|---|
| `WorkflowExecutionIntentSelector` | `Select`: which of the workflow descriptors the request asks for, none if it asks for none |
| `WorkflowExecutionIntentParameterExtractor` | `ExtractParameters`: given **only the selected descriptors**, schema included, a `WorkflowExecutionIntent` (workflow ID and `Params`) for each |

A classifier that does both in one pass implements `IntentClassifier` directly.

## What an intent is

An `Intent` is the command or the query a request was turned into: a list of `WorkflowExecutionIntent{WorkflowID, Params}`, one for each workflow the request asks for, possibly several (*"restart api and tell me the version"*). An `Intent` with none is a **reasoning intent** (`IsReasoningIntent`): nothing has to run, and the request goes to the `ReasoningEngine`.

A workflow that cannot run for a reason the user should hear of (a missing parameter, an invalid input) is not an error: its `WorkflowExecutionResult` has a `Status` that says so and the reason in `Data`, and the `WorkflowExecutionSynthesizer` explains it. A workflow that runs asynchronously returns its `ExecutionID` for the host to follow up.

## How a request flows

```text
CompletionRequest
      │
      ▼
1. WorkflowDescriptors()     the workflows: ID, name, description, input schema
      │
      ├─ none ──────────────────────────────────────────────┐
      ▼                                                     │
2. Classify()                which workflows, if any, and their parameters
      │                                                     │
      ├─ reasoning intent (none) ───────────────────────────┤
      ▼                                                     ▼
3. ExecuteWorkflows()                              5. Reason()
      │
      ▼
4. Synthesize()  ─▶  CompletionResponse                     ─▶  CompletionResponse
```

`ControlPlane.Process(ctx, req)`:

1. **Describe.** `WorkflowDescriptors` lists the workflows: an ID, a name, a description, and the JSON Schema of their parameters (`InputSchema`).
2. **Classify.** Unless there are no workflows, `Classify` is given the request and the descriptors, and answers an `Intent`.
3. **Run.** If the `Intent` names workflows, `ExecuteWorkflows` runs them, all in one call, and answers a `WorkflowExecutionResult` for each.
4. **Synthesize.** `WorkflowExecutionSynthesizer.Synthesize` is given a `WorkflowExecutionSynthesis` (the request, the intent and the results) and returns the `CompletionResponse`.
5. **Or reason.** For a reasoning intent, `ReasoningEngine.Reason` is given the request and the descriptors and returns the `CompletionResponse`. Nothing is synthesized.

A `CompletionResponse` carries a status, a content type and a body: Proxias never looks into it.


## Quickstart

A runnable hello world is in [examples/basic](examples/basic): one workflow, with Claude as the classifier and the synthesizer through the [Anthropic adapter](#anthropic-adapter), and a stub as the reasoning engine. It needs `ANTHROPIC_API_KEY`: `go run ./examples/basic`.

### Embedding in a Server: intercepting HTTP requests

Proxias is not a server: the transport stays the host's. The `openai` package adapts the OpenAI chat completions API: `openai.DecodeRequest` turns a request body into a `proxias.CompletionRequest`, and `openai.Handler` serves a `ControlPlane` over `net/http`. With it, a server answers what its workflows can and forwards the rest to the upstream LLM.

```go
controlPlane, err := proxias.NewControlPlane(
	proxias.NewCompositeIntentClassifier(selector, extractor),
	workflows,
	proxias.WorkflowExecutionSynthesizerFunc(
		func(ctx context.Context, s proxias.WorkflowExecutionSynthesis) (proxias.CompletionResponse, error) {
			return upstream.Summarize(ctx, s) // one LLM pass over the workflows' results
		}),
	reasoner, // forwards the request to the upstream LLM, byte for byte (CompletionRequest.Request)
)
if err != nil {
	return err
}

mux := http.NewServeMux()
mux.Handle("POST /v1/chat/completions", openai.Handler(controlPlane))
```

The handler answers 400 for a body that is not a chat completions request (413 if it is too large), and 502, without saying why, when the control plane fails: log that error yourself, for instance with a `ControlPlane` that decorates yours. A response body is flushed as it is read, so a stream reaches the client as it is produced. To write your own handler, use `openai.DecodeRequest`.

### Embedding in an agent implementation: local admin voice and chat workflows

The agent answers the host application's admin commands, typed or spoken, without an LLM round trip for each.

```go
controlPlane, err := proxias.NewControlPlane(adminCommands, adminWorkflows, adminSynthesizer, assistant)

response, err := controlPlane.Process(ctx, proxias.CompletionRequest{Messages: []proxias.CompletionMessage{
	{Role: proxias.CompletionMessageRoleUser, Content: []byte(transcript)}, // from the chat box or speech-to-text
}})
```

Free-form questions still reach the LLM, through `assistant`, the `ReasoningEngine`.

## Anthropic adapter

`adapters/anthropic` makes Claude act as the fast, system-1 part of the flow: it reads the conversation and answers with a small JSON object, the intent. No tool is involved. It uses the official [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-go): you build its client, so the key, the timeout, the retries and the base URL are yours, and so is the model.

```go
client := anthropicsdk.NewClient(option.WithAPIKey(key), option.WithMaxRetries(0))

// The whole classification in one call.
classifier := anthropic.NewIntentClassifier(client, anthropicsdk.ModelClaudeHaiku4_5)

// Or the two stages of a CompositeIntentClassifier, each with its own model.
classifier = proxias.NewCompositeIntentClassifier(
    anthropic.NewWorkflowExecutionIntentSelector(client, anthropicsdk.ModelClaudeHaiku4_5),
    anthropic.NewWorkflowExecutionIntentParameterExtractor(client, anthropicsdk.ModelClaudeSonnet4_5),
)

// The reply, wrapped in the wire format of your clients.
synthesizer, err := anthropic.NewWorkflowExecutionSynthesizer(client, anthropicsdk.ModelClaudeSonnet4_5, encode) // an error if encode is nil
```

A failing call returns the SDK's error, wrapped: `errors.As` an `*anthropicsdk.Error` for its status code. An answer that is not the expected JSON is an error too: it is for the host to decide whether to fall back to the `ReasoningEngine`.

## Performance

`go test -bench .` measures the ControlPlane's own overhead, with fakes that answer instantly: about 0.1 µs and no allocation per request. The time that counts is your classifier's and your extractor's: keep it local and it adds milliseconds at most, far under the 50 ms a request should take before the LLM is reached.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). Apache License 2.0, see [LICENSE](LICENSE).
