<p align="center"><img src="logo.svg" alt="Proxias" width="120"></p>

# Proxias

Deterministic intent interception for AI agents.

Proxias sits in front of an LLM. An **IntentClassifier** names what a message asks for, the **Interceptor** runs the compiled, deterministic **Workflows** it asks for, and only a message with no intent goes to the **Fallback**, usually an LLM, which then answers from the workflows' results. No multi-turn tool loop for what code can do reliably, quickly and cheaply.

Standard library only; it embeds in any Go program and knows nothing of your application.

## Install

```sh
go get github.com/morebec/proxias
```

## Example

```go
interceptor := proxias.NewInterceptor(
	proxias.WithIntentClassifier(rules.RunCommand{}), // or your own: a local rule, a model, a service
	proxias.WithFallback(proxias.FallbackFunc(func(ctx context.Context, req proxias.Request, results []proxias.Result) (any, error) {
		return askTheLLM(ctx, req, results) // gets the results of the workflows that ran, if any
	})),
)

// A workflow is a typed Go function: JSON in, JSON out (and an optional UI spec).
interceptor.Register(proxias.Typed("app_status", "App status", "Reports the version and health",
	func(ctx context.Context, _ struct{}) (AppStatus, error) { return currentStatus(), nil }))

outcome, err := interceptor.Intercept(ctx, proxias.Request{
	Messages: []proxias.Message{{Role: proxias.RoleUser, Content: "/run app_status"}},
	Meta:     map[string]any{"user": "betty"}, // yours: passed untouched to the classifier and the fallback
})
// outcome.Results: what the workflows answered; outcome.FellBack / outcome.Completion: the fallback's answer.
```

A runnable version is in [examples/admin](examples/admin).

## How a request flows

1. Only a last message from the **user** is classified. After a tool result, or with an empty conversation, the host's own loop continues: straight to the fallback.
2. A classifier that **fails** is as if it found no intent: it never blocks an answer (`Hooks` and the logger tell why).
3. The intent's workflows run **in order**. One that fails ends the series, its failure being its result. If the first is not registered, it is as if no intent was found.
4. The **fallback** answers from the results. Without a fallback, the results are the answer (and `ErrNoFallback` when there are none).

## Performance

`go test -bench .` measures the Interceptor's own overhead (a fake classifier, one workflow): about 20 µs and 20 allocations per request, far under 50 ms. A classifier that calls a network (a model, a service) adds its own latency, which Proxias does not control; a local one such as `rules` adds microseconds.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). Apache License 2.0, see [LICENSE](LICENSE).
