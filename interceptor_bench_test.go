package proxias_test

import (
	"context"
	"testing"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/intentclassifiertest"
)

// The overhead of the Interceptor itself (classifying with a fake, running a workflow, no fallback). A
// classifier that calls a network adds its own latency, which Proxias does not control.
func BenchmarkIntercept(b *testing.B) {
	interceptor := proxias.NewInterceptor(proxias.WithIntentClassifier(&intentclassifiertest.Fixed{Intent: proxias.Intent{WorkflowIDs: []string{"hello"}}}))
	interceptor.Register(hello())
	request := user("bob")
	b.ReportAllocs()
	for b.Loop() {
		if _, err := interceptor.Intercept(context.Background(), request); err != nil {
			b.Fatal(err)
		}
	}
}
