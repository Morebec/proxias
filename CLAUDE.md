# CLAUDE.md

Proxias is a **Go SDK** (module `github.com/morebec/proxias`): a deterministic execution plane for LLM applications. A `ControlPlane` turns a request into workflow executions (or a reasoning pass) and the results back into a response. See [README.md](README.md) for the user-facing description and [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor rules.

This file describes the code **as it is**. Anything planned but not built is under [Not implemented](#not-implemented); do not treat it as existing.

## Project rules

- **It is a library.** Every exported identifier is public API, versioned by [Release Please](https://github.com/googleapis/release-please). An exported change is a breaking change: say so in the commit (`feat!:` / `BREAKING CHANGE:`).
- **Knows nothing of any host application**: no application types, config or globals. Transports live in the host; the request is the typed model, with its unnamed fields in `Extra`: marshalling it gives the body that a host forwards. `http.go` is the one transport (the OpenAI chat completions endpoint as an `http.Handler`). **The model of the root package is the OpenAI chat completions API** (`ChatCompletionRequest`, `ChatMessage`, `ChatCompletion`...): it carries everything a harness sends (images, tool calls, cache breakpoints, streaming) so that a reasoning engine can be a transparent proxy, and an adapter maps to and from it at its edge; do not grow it into a server or add other transports. `adapters/anthropic` is a model adapter: it implements the components with Claude through the official `anthropic-sdk-go` (the only dependency of the library besides the tests' testify): do not reimplement what the SDK does (HTTP, auth, retries, errors).
- No `main` at the root, no `init()`, no package-level mutable state: a host may embed several ControlPlanes in one process, and a ControlPlane is used concurrently.
- **Never log a request**: not its messages, nor its headers (they carry credentials), nor the parameters or results of a workflow. (There is no logging yet, see [Not implemented](#not-implemented).)

## Layout

Follows [golang-standards/project-layout](https://github.com/golang-standards/project-layout), adapted to an SDK.

| Path | What |
|---|---|
| `/` (module root) | Package `proxias`, the public API: the library is the root package, importable as `github.com/morebec/proxias`. `classification.go` (`IntentClassifier`, `Intent`, `CompositeIntentClassifier`, its selector and parameter extractor stages), `workflowexecution.go` (`WorkflowExecutionEngine`, `WorkflowExecutionSynthesizer`, descriptors, results), `reasoning.go` (`ReasoningEngine`), `completion_req.go` (the request side of the OpenAI chat completions model: `ChatCompletionRequest`, `ChatMessage`, `ChatContent` parts, tools; each keeps its unnamed JSON fields in `Extra`, see `completion_extra.go`) and `http.go` (the HTTP side: `DecodeChatCompletionRequest(*http.Request)` reads a request, and a copy of the headers in `Header`; `WriteChatCompletionResponse(w, resp)` writes a response, headers, status, body flushed as read, body closed; `NewChatCompletionHandler(controlPlane)` serves a ControlPlane as the OpenAI chat completions endpoint, the one transport: that API is the model of Proxias, so it is not an adapter) and `completion_resp.go` (the response side: `ChatCompletionResponse`, the HTTP envelope; `ChatCompletion` and `ChatCompletionChunk`, the bodies; `NewChatCompletionResponse`, `NewChatCompletionStream`), `control_plane.go` (the `ControlPlane`, the chain of `Middleware` it runs), `middleware.go` (`Middleware`, `ProcessFunc`, the workflow descriptors in the context), `workflow_interception_middleware.go` (`NewWorkflowInterceptionMiddleware`: classify, run, synthesize, or pass on), `reasoning_middleware.go` (`NewReasoningMiddleware`: the last step, which answers), `prompt.go` (`Prompt`: builds an LLM prompt from paragraphs, `## ` sections, bullet points and labeled bullet points; nothing in the repo uses it since `adapters/anthropic` moved to constant prompts), `doc.go`, and the tests. |
| `adapters/anthropic/` | Package `anthropic` (package comment in `reasoning.go`): `IntentClassifier` (`classifier.go`: one call, the conversation and the workflows as JSON, answering the executions; no workflow means no call), `WorkflowExecutionIntentSelector` (`classifier_selector.go`: answers the IDs of the workflows, without their schemas) and `WorkflowExecutionIntentParameterExtractor` (`classifier_extractor.go`: answers the executions of the matched workflows, with their schemas; its decoding is its own, duplicated from the classifier's on purpose, so each evolves alone), and `ReasoningEngine`. The components embed `claude` (`classifier.go`: the SDK client, the model and `askJSON`, the one call they make); the decisions are each component's own, built from the host's `anthropic.Client` of the SDK (the host owns the key, timeout, retries, base URL) and a model. `request.go` maps the chat completion request to Messages API params (system/developer to the system blocks, tool calls and results, images, `cache_control`, tools, stop, max tokens; consecutive messages of one role are merged; an unknown role is an error), `response.go` maps the answer (`chatCompletion`) and the stream events (`chunks`) back. Without `req.Stream` it calls `Messages.New`; with it `NewStreaming`, reading the first event before answering, so an API error is a returned error and not a stream. Errors wrap the SDK's (`errors.As` an `*anthropic.Error`). Thinking is not returned. `WorkflowExecutionSynthesizer` (`synthesizer.go`: one call that writes the reply from the conversation and the results, as a chat completion or, when the request streams, as a stream of one whole reply). Depends on the root package only. Tests use an `httptest` server (`mock` in `reasoning_test.go`). `adapters/anthropic___/` is a backup of an earlier version, which does not compile. |
| `examples/basic/` | Runnable example, a `package main`. One directory per example. |
| `assets/` | Logo. |
| `.github/workflows/` | `ci.yml`, `pr-title.yml`, `release-please.yml`. |

Not present, create only when needed: `/docs` (design docs beyond the README), `/internal` (private code), `/cmd` (a binary, which must stay thin). A new adapter goes in `adapters/<name>`; any other new public package gets its own top-level directory. The root holds only the `proxias` package. No build output, vendored code or `.DS_Store` in the repo (`coverage.out` is a local artifact of `task test:cover`; do not commit it).

## Commands

Tasks run with [Task](https://taskfile.dev) (`Taskfile.yml`; `task --list` shows all).

| Task | Does |
|---|---|
| `task check` | What CI checks, in order: `fmt:check` (gofmt), `lint`, `test`, `bench:smoke`. Run before every push. |
| `task test` | `go test -race ./...` |
| `task test:cover` | Tests with coverage, writes `coverage.out` |
| `task lint` | golangci-lint at the version pinned in `Taskfile.yml` (keep it equal to `ci.yml`'s) |
| `task fmt` | Format with the formatters of `.golangci.yml` |
| `task bench` / `task bench:smoke` | The ControlPlane's own overhead / 100 runs to see it still runs |
| `task example` | Run `./examples/basic` |
| `task sandbox` | Run the middleware sandbox (`examples/sandbox`) on :8787, the endpoint a real harness talks to |
| `task sandbox:openclaw` | Run OpenClaw in a Docker container, configured to use the sandbox (fresh state in `.sandbox/`) |
| `task sandbox:openclaw:agent -- "message"` | Send a message to that OpenClaw, through its gateway |
| `task sandbox:openclaw:dashboard` | Print the link to that OpenClaw's Control UI, with its token |
| `task deadcode` | Functions no test or example reaches |

Lint rules are in `.golangci.yml`. Fix a finding; do not silence it. A `//nolint` needs the linter's name and a reason (`nolintlint` enforces it).

## Architecture

### The four interfaces

One per step, no more:

| Interface | Role |
|---|---|
| `IntentClassifier` | Which workflows a request asks for, and their parameters (`Intent`). |
| `WorkflowExecutionEngine` | Describes the workflows (`WorkflowDescriptors`) and runs them (`ExecuteWorkflows`). |
| `WorkflowExecutionSynthesizer` | The reply to what ran. |
| `ReasoningEngine` | The reply when nothing ran. |

Each has a `...Func` adapter except `WorkflowExecutionEngine` (two methods). Parameter extraction is not a fifth interface: it is the second stage of `CompositeIntentClassifier` (`WorkflowExecutionIntentSelector`, then `WorkflowExecutionIntentParameterExtractor`). The four components are given to middleware: `NewWorkflowInterceptionMiddleware` takes the classifier, the engine and the synthesizer, and `NewReasoningMiddleware` takes the reasoning engine. Constructors **return an error** for a nil component: a library does not panic in its constructors. `ControlPlane` is an interface; its implementation is unexported.

### The chain: `Process(ctx, req)`

`NewControlPlane(middlewares...)` builds the chain once, in the order given: the first middleware sees a request first. `Process` checks `ctx.Err()`, then runs the chain. The first step that answers bypasses the rest; a step that does not answer calls `next`, with the request it wants the rest to see and a context it may have augmented. The usual chain is:

1. `WorkflowInterceptionMiddleware`: `WorkflowDescriptors` lists the workflows. If there are none, or `Classify` finds no workflow, it calls `next` with the descriptors in the context (`WithWorkflowDescriptors`). Otherwise `ExecuteWorkflows` runs all the executions in one call, the step checks there is one result per execution, checks `ctx.Err()` again, then `Synthesize` gets a `WorkflowExecutionSynthesis` (request, intent, results) and answers.
2. `ReasoningMiddleware`: the last step. It always answers: `ReasoningEngine.Reason` is given the request and the descriptors of the context.

If no step answers, `Process` returns an error.

### Behavior rules

- **Middleware rewrite requests by copy.** A request is passed by value: a step changes its own copy and calls `next`. Its slices, maps and `Extra` are shared with the caller, so a step **replaces** them (`req.Messages = append([]ChatMessage{system}, req.Messages...)`), never edits them in place. The request is the source of truth: nothing stores a copy of its JSON, so a rewrite is what gets sent. Fields that the model does not name are kept in `Extra`, and written back.
- **Context carries data about the request, the request carries the request.** A step that needs to tell the rest something (the workflows a request may ask for) puts it in the context, with an unexported key.
- **The engine owns the workflows.** The control plane does not check the chosen workflows against the descriptors: the engine answers an unknown workflow with a failed result. It must return exactly one result per execution (the control plane checks the count, not the order).
- `CompositeIntentClassifier`'s extractor gets only the selected workflows' descriptors, and is not called when none was selected.
- `Intent` carries nothing classifier-specific (confidence, score), and the ControlPlane must not depend on it.
- **Failure rules** (documented behavior, keep the doc comments in step): errors of `WorkflowDescriptors`, `Classify`, `ExecuteWorkflows`, `Synthesize` and `Reason` are returned wrapped, with the stage and the `proxias:` prefix (`fmt.Errorf("proxias: execute workflows: %w", err)`) so `errors.Is` finds the cause; there are no sentinel errors. A failure the user should hear of is a result's `Status` and `Data`, not an error. A `ctx` error is returned as is.
- **No `recover`.** A host's panic is the host's to recover; `Process` lets it through (`TestControlPlaneProcessDoesNotRecoverPanics`).
- Roles are not integrated: the chain does not look at them, so a request ending with a tool result is classified like any other. `ChatMessage.HasToolRole` is for hosts.

### Not implemented

Earlier versions of this file and of the README mentioned some of these; they are not in the code, and the docs now follow the code.

- **Logging is middleware, not a feature of the library.** The library does not log. A host logs with a middleware, as `examples/sandbox` does (`logRequests`): what a request costs and how it ends, never its messages, which are data. Decorators are for the few cases a middleware cannot cover.
- No `/docs`, `/internal`, `/cmd`.

## Code conventions

- **Conventional Commits** for commits and PR titles (`feat: ...`, `fix(engine): ...`, `chore: ...`). PRs are squash-merged, so the title is the changelog line; CI enforces it. Do not edit `.release-please-manifest.json` or tag by hand.
- Every exported identifier has a doc comment starting with its name. Each package has a package comment (`doc.go` at the root, `client.go` in `adapters/anthropic`).
- Comments say why, not what; match the surrounding density.
- Keep the ControlPlane's own overhead under a microsecond (README claims about 0.1 µs, no allocation): run `task bench` when touching `control_plane.go`.

## Testing conventions

- **External tests** (`package proxias_test`, `package anthropic_test`): they test what a host sees. Table-driven.
- **Naming:** a test is named after the type and the method (`TestChatCompletionRequestLastMessage`, `TestCompositeIntentClassifierClassify`), then what it asserts when a method needs several tests (`TestControlPlaneProcessReturnsContextErrorBeforeProcessing`). One concern per test function.
- **Given / when / then.** `given` is what is injected (the dependencies and options), `when` the arguments of the method under test, `then` what is expected. Case names say it: `GIVEN ... WHEN ... THEN ...`, leaving out a clause that does not vary; the plain everything-goes-right case is named `canonical`. A table case has `name`, `given`, `when` (omitted when the arguments never vary), `then`, **one field per line, never one-lined**. A clause with several values is a struct local to the test: `assumptions`, `args`, `expect`. Range with `for _, tt := range tests`. A test that is not a table marks its sections `// Given`, `// When`, `// Then`.
- Prefer typed assertion funcs where one says more than a value (`then assert.ErrorAssertionFunc`, `then assert.BoolAssertionFunc`).
- **Fakes** live in `control_plane_test.go` and `classification_test.go` and are shared by the package's tests:
  - `mock*` parameterized helpers, not a bespoke fake per case: `mockClassifier(intent, err)`, `mockWorkflowExecutionSynthesizer(resp, err)`, `mockReasoningEngine(resp, err)`, `mockWorkflowSelector(...)`, `mockParamExtractor(...)`, and the struct `mockWorkflowExecutionEngine`.
  - `recording*` when a test must assert on what a stage was given (`recordingReasoningEngine`, `recordingWorkflowExecutionSynthesizer`): a fake records its input, it does not encode it in unrelated output fields.
  - `some*` fixtures (`someRequest`, `someWorkflowDescriptor`).
  - Fill nothing a test does not assert (no `Content` on a message whose role is what is tested).
- Add a case for each behavior in the README's flow.
