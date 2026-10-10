# CLAUDE.md

Proxias is a **Go SDK** (module `github.com/morebec/proxias`): a deterministic execution plane for LLM applications. A `ControlPlane` turns a request into workflow executions (or a reasoning pass) and the results back into a response. See [README.md](README.md) for the user-facing description and [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor rules.

This file describes the code **as it is**. Anything planned but not built is under [Not implemented](#not-implemented); do not treat it as existing.

## Project rules

- **It is a library.** Every exported identifier is public API, versioned by [Release Please](https://github.com/googleapis/release-please). An exported change is a breaking change: say so in the commit (`feat!:` / `BREAKING CHANGE:`).
- **Knows nothing of any host application**: no application types, config or globals. Transports live in the host; `CompletionRequest.Request` carries the raw request for a host that needs it. `adapters/openai` is the one transport adapter (OpenAI chat completions: `DecodeRequest` and an `http.Handler`); do not grow it into a server or add other transports. `adapters/anthropic` is a model adapter: it implements the components with Claude through the official `anthropic-sdk-go` (the only dependency of the library besides the tests' testify): do not reimplement what the SDK does (HTTP, auth, retries, errors).
- No `main` at the root, no `init()`, no package-level mutable state: a host may embed several ControlPlanes in one process, and a ControlPlane is used concurrently.
- **Never log a request**: not its messages, nor the parameters or results of a workflow. (There is no logging yet, see [Not implemented](#not-implemented).)

## Layout

Follows [golang-standards/project-layout](https://github.com/golang-standards/project-layout), adapted to an SDK.

| Path | What |
|---|---|
| `/` (module root) | Package `proxias`, the public API: the library is the root package, importable as `github.com/morebec/proxias`. `classification.go` (`IntentClassifier`, `Intent`, `CompositeIntentClassifier`, its selector and parameter extractor stages), `workflowexecution.go` (`WorkflowExecutionEngine`, `WorkflowExecutionSynthesizer`, descriptors, results), `reasoning.go` (`ReasoningEngine`), `completion.go` (`CompletionRequest`, `CompletionMessage`, `CompletionResponse`), `control_plane.go` (the orchestrator), `control_plane_options.go` (`ControlPlaneOption`), `prompt.go` (`Prompt`: builds an LLM prompt from paragraphs, `## ` sections, bullet points and labeled bullet points; nothing in the repo uses it since `adapters/anthropic` moved to constant prompts), `doc.go`, and the tests. |
| `adapters/openai/` | Package `openai`: wire types, `DecodeRequest`, `Handler`. Depends on the root package, never the reverse. Its package comment is in `openai.go` (no `doc.go`). |
| `adapters/anthropic/` | Package `anthropic` (package comment in `client.go`): four components built from the host's `anthropic.Client` of the SDK (the host owns the key, timeout, retries, base URL) and a **required model** (`New...(client, model)`; no default). The classification components make Claude a system-1 decision model: a static, cacheable system prompt (a constant per component), a user turn that is the JSON of the conversation (`messages`) and of the workflows (`WorkflowDescriptor`'s JSON; the selector's without schemas), and a JSON answer parsed leniently. **No tool use.** `IntentClassifier` answers `{"executions":[{"workflow_id","params"}]}` (none = reasoning intent), `WorkflowExecutionIntentSelector` `{"workflows":[ids]}`, `WorkflowExecutionIntentParameterExtractor` the executions of the matched workflows; an ID that was not offered is dropped. `WorkflowExecutionSynthesizer` writes text, wrapped by a required `ResponseEncoder`. Errors wrap the SDK's (`errors.As` an `*anthropic.Error`; its text includes the API's message, which can echo the request); an unreadable answer is an error, the host decides on a fallback. No confidence threshold, no streaming. Internals: `claude` in `client.go` (the SDK client and model; `ask`, `askJSON`), `classification.go`, `synthesis.go`. Tests use an `httptest` server (`mock_test.go`). |
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

Each has a `...Func` adapter except `WorkflowExecutionEngine` (two methods). Parameter extraction is not a fifth interface: it is the second stage of `CompositeIntentClassifier` (`WorkflowExecutionIntentSelector`, then `WorkflowExecutionIntentParameterExtractor`). Dependencies are `NewControlPlane`'s arguments, which **returns an error if any of the four is nil**: a library does not panic in its constructors. `ControlPlane` is an interface; its implementation is unexported.

### Flow of `Process(ctx, req)`

1. `ctx.Err()` is checked.
2. `WorkflowDescriptors` lists the workflows.
3. If there are none, the classifier is **not** called and the intent is a reasoning intent. Otherwise `Classify` is given the descriptors with their `InputSchema`.
4. A reasoning intent (`Intent.IsReasoningIntent`: no `WorkflowExecutions`) goes to `Reason`, which gets the request and the descriptors.
5. Otherwise `ExecuteWorkflows` runs all the executions in one call, the control plane checks there is one result per execution, checks `ctx.Err()` again, then `Synthesize` gets a `WorkflowExecutionSynthesis` (request, intent, results).

### Behavior rules

- **The engine owns the workflows.** The control plane does not check the chosen workflows against the descriptors: the engine answers an unknown workflow with a failed result. It must return exactly one result per execution (the control plane checks the count, not the order).
- `CompositeIntentClassifier`'s extractor gets only the selected workflows' descriptors, and is not called when none was selected.
- `Intent` carries nothing classifier-specific (confidence, score), and the ControlPlane must not depend on it.
- **Failure rules** (documented behavior, keep the doc comments in step): errors of `WorkflowDescriptors`, `Classify`, `ExecuteWorkflows`, `Synthesize` and `Reason` are returned wrapped, with the stage and the `proxias:` prefix (`fmt.Errorf("proxias: execute workflows: %w", err)`) so `errors.Is` finds the cause; there are no sentinel errors. A failure the user should hear of is a result's `Status` and `Data`, not an error. A `ctx` error is returned as is.
- **No `recover`.** A host's panic is the host's to recover; `Process` lets it through (`TestControlPlaneProcessDoesNotRecoverPanics`).
- Roles are not integrated: `Process` does not look at them, so a request ending with a tool result is classified like any other. `CompletionMessage.HasToolRole` is for hosts.

### Not implemented

Earlier versions of this file and of the README mentioned some of these; they are not in the code, and the docs now follow the code.

- **Logging.** There is no `logging.go`, no logger decorators, no `WithLogger`. `ControlPlaneOption` is `func(*controlPlane)` with no option defined, and `NewControlPlane` applies the options before assigning the components. If logging is added: decorators of the four components applied by `NewControlPlane`, configured by `WithLogger`, at Debug level, once a stage is done, never logging an error that is also returned.
- No `/docs`, `/internal`, `/cmd`.

## Code conventions

- **Conventional Commits** for commits and PR titles (`feat: ...`, `fix(engine): ...`, `chore: ...`). PRs are squash-merged, so the title is the changelog line; CI enforces it. Do not edit `.release-please-manifest.json` or tag by hand.
- Every exported identifier has a doc comment starting with its name. Each package has a package comment (`doc.go` at the root, `openai.go` in `adapters/openai`, `client.go` in `adapters/anthropic`).
- Comments say why, not what; match the surrounding density.
- Keep the ControlPlane's own overhead under a microsecond (README claims about 0.1 µs, no allocation): run `task bench` when touching `control_plane.go`.

## Testing conventions

- **External tests** (`package proxias_test`, `package openai_test`, `package anthropic_test`): they test what a host sees. Table-driven.
- **Naming:** a test is named after the type and the method (`TestCompletionRequestLastMessage`, `TestCompositeIntentClassifierClassify`), then what it asserts when a method needs several tests (`TestControlPlaneProcessReturnsContextErrorBeforeProcessing`). One concern per test function.
- **Given / when / then.** `given` is what is injected (the dependencies and options), `when` the arguments of the method under test, `then` what is expected. Case names say it: `GIVEN ... WHEN ... THEN ...`, leaving out a clause that does not vary; the plain everything-goes-right case is named `canonical`. A table case has `name`, `given`, `when` (omitted when the arguments never vary), `then`, **one field per line, never one-lined**. A clause with several values is a struct local to the test: `assumptions`, `args`, `expect`. Range with `for _, tt := range tests`. A test that is not a table marks its sections `// Given`, `// When`, `// Then`.
- Prefer typed assertion funcs where one says more than a value (`then assert.ErrorAssertionFunc`, `then assert.BoolAssertionFunc`).
- **Fakes** live in `control_plane_test.go` and `classification_test.go` and are shared by the package's tests:
  - `mock*` parameterized helpers, not a bespoke fake per case: `mockClassifier(intent, err)`, `mockWorkflowExecutionSynthesizer(resp, err)`, `mockReasoningEngine(resp, err)`, `mockWorkflowSelector(...)`, `mockParamExtractor(...)`, and the struct `mockWorkflowExecutionEngine`.
  - `recording*` when a test must assert on what a stage was given (`recordingReasoningEngine`, `recordingWorkflowExecutionSynthesizer`): a fake records its input, it does not encode it in unrelated output fields.
  - `some*` fixtures (`someRequest`, `someWorkflowDescriptor`).
  - Fill nothing a test does not assert (no `Content` on a message whose role is what is tested).
- Add a case for each behavior in the README's flow.
