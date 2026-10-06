// Command admin shows how a host, such as an assistant's admin panel, answers its commands deterministically
// and leaves everything else to an LLM (here a stand-in).
package main

import (
	"context"
	"fmt"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/rules"
)

type noInput struct{}

type appStatus struct {
	Version string `json:"version"`
	Healthy bool   `json:"healthy"`
}

func main() {
	interceptor := proxias.NewInterceptor(
		proxias.WithIntentClassifier(rules.RunCommand{}),
		proxias.WithFallback(proxias.FallbackFunc(func(_ context.Context, _ proxias.Request, _ []proxias.Result) (any, error) {
			return "(the LLM answers)", nil
		})),
	)
	interceptor.Register(proxias.Typed("app_status", "App status", "Reports the version and health of the app",
		func(context.Context, noInput) (appStatus, error) {
			return appStatus{Version: "1.2.3", Healthy: true}, nil
		}))

	for _, text := range []string{"/run app_status", "what can you do?"} {
		outcome, err := interceptor.Intercept(context.Background(), proxias.Request{
			Messages: []proxias.Message{{Role: proxias.RoleUser, Content: text}},
		})
		if err != nil {
			panic(err)
		}
		for _, result := range outcome.Results {
			fmt.Printf("%-18s workflow %s: %s\n", text, result.WorkflowID, result.Output)
		}
		if outcome.FellBack {
			fmt.Printf("%-18s fallback: %v\n", text, outcome.Completion)
		}
	}
}
