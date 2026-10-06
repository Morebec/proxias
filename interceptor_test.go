package proxias_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/intentclassifiertest"
)

type greeting struct{ Message string }

type greeted struct {
	Text string `json:"text"`
}

func (greeted) UISpec() json.RawMessage { return json.RawMessage(`{"type": "card"}`) }

func hello() proxias.Workflow {
	return proxias.Typed("hello", "Hello", "greets", func(_ context.Context, in greeting) (greeted, error) {
		return greeted{Text: "hello " + in.Message}, nil
	})
}

func broken(id string) proxias.Workflow {
	return proxias.Typed(id, id, "", func(context.Context, greeting) (greeted, error) { return greeted{}, errors.New("boom") })
}

func user(text string) proxias.Request {
	return proxias.Request{Messages: []proxias.Message{{Role: proxias.RoleUser, Content: text}}}
}

// fallbackSpy records what the Fallback was given.
type fallbackSpy struct {
	called  bool
	results []proxias.Result
}

func (f *fallbackSpy) Complete(_ context.Context, _ proxias.Request, results []proxias.Result) (any, error) {
	f.called, f.results = true, results
	return "llm", nil
}

func TestInterceptor(t *testing.T) {
	tests := []struct {
		name         string
		request      proxias.Request
		classifier   *intentclassifiertest.Fixed
		withFallback bool
		wantErr      error
		wantFellBack bool
		wantResults  int
		wantFailed   bool
		wantCalls    int
	}{
		{name: "an intent runs its workflow, the fallback answers from the results", request: user("bob"),
			classifier:   &intentclassifiertest.Fixed{Intent: proxias.Intent{WorkflowIDs: []string{"hello"}}},
			withFallback: true, wantFellBack: true, wantResults: 1, wantCalls: 1},
		{name: "without a fallback the results are the answer", request: user("bob"),
			classifier:  &intentclassifiertest.Fixed{Intent: proxias.Intent{WorkflowIDs: []string{"hello"}}},
			wantResults: 1, wantCalls: 1},
		{name: "no intent goes to the fallback", request: user("bob"), classifier: &intentclassifiertest.Fixed{},
			withFallback: true, wantFellBack: true, wantCalls: 1},
		{name: "no intent and no fallback", request: user("bob"), classifier: &intentclassifiertest.Fixed{},
			wantErr: proxias.ErrNoFallback, wantCalls: 1},
		{name: "a failing classifier never blocks the answer", request: user("bob"),
			classifier:   &intentclassifiertest.Fixed{Err: errors.New("down")},
			withFallback: true, wantFellBack: true, wantCalls: 1},
		{name: "a tool result is not classified", request: proxias.Request{Messages: []proxias.Message{{Role: proxias.RoleTool, Content: "x"}}},
			classifier:   &intentclassifiertest.Fixed{Intent: proxias.Intent{WorkflowIDs: []string{"hello"}}},
			withFallback: true, wantFellBack: true},
		{name: "an empty conversation is not classified", request: proxias.Request{},
			classifier:   &intentclassifiertest.Fixed{Intent: proxias.Intent{WorkflowIDs: []string{"hello"}}},
			withFallback: true, wantFellBack: true},
		{name: "an unknown first workflow is as if none was asked", request: user("bob"),
			classifier:   &intentclassifiertest.Fixed{Intent: proxias.Intent{WorkflowIDs: []string{"ghost"}}},
			withFallback: true, wantFellBack: true, wantCalls: 1},
		{name: "a failing workflow ends the series, its failure is its result", request: user("bob"),
			classifier:  &intentclassifiertest.Fixed{Intent: proxias.Intent{WorkflowIDs: []string{"bad", "hello"}}},
			wantResults: 1, wantFailed: true, wantCalls: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spy := &fallbackSpy{}
			options := []proxias.Option{proxias.WithIntentClassifier(tt.classifier)}
			if tt.withFallback {
				options = append(options, proxias.WithFallback(spy))
			}
			interceptor := proxias.NewInterceptor(options...)
			interceptor.Register(hello())
			interceptor.Register(broken("bad"))

			outcome, err := interceptor.Intercept(context.Background(), tt.request)

			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("error = %v, want %v", err, tt.wantErr)
			}
			if outcome.FellBack != tt.wantFellBack || len(outcome.Results) != tt.wantResults {
				t.Fatalf("fellBack=%v results=%d, want %v and %d", outcome.FellBack, len(outcome.Results), tt.wantFellBack, tt.wantResults)
			}
			if tt.wantResults > 0 && outcome.Results[0].Failed != tt.wantFailed {
				t.Errorf("failed = %v, want %v", outcome.Results[0].Failed, tt.wantFailed)
			}
			if tt.wantFellBack && len(spy.results) != tt.wantResults {
				t.Errorf("the fallback got %d results, want %d", len(spy.results), tt.wantResults)
			}
			if tt.classifier.Calls != tt.wantCalls {
				t.Errorf("classifier calls = %d, want %d", tt.classifier.Calls, tt.wantCalls)
			}
		})
	}
}

func TestTypedWorkflow(t *testing.T) {
	interceptor := proxias.NewInterceptor(proxias.WithIntentClassifier(&intentclassifiertest.Fixed{Intent: proxias.Intent{WorkflowIDs: []string{"hello"}}}))
	interceptor.Register(hello())

	outcome, err := interceptor.Intercept(context.Background(), user("bob"))
	if err != nil {
		t.Fatal(err)
	}
	result := outcome.Results[0]
	if string(result.Output) != `{"text":"hello bob"}` || string(result.UI) != `{"type": "card"}` {
		t.Errorf("output = %s, ui = %s", result.Output, result.UI)
	}
}

func TestCandidatesKeepRegistrationOrder(t *testing.T) {
	interceptor := proxias.NewInterceptor()
	interceptor.Register(broken("b"))
	interceptor.Register(hello())
	interceptor.Register(broken("b")) // replaced, keeps its place

	candidates := interceptor.Candidates()
	if len(candidates) != 2 || candidates[0].ID != "b" || candidates[1].ID != "hello" {
		t.Errorf("candidates = %v", candidates)
	}
}
