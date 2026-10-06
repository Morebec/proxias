package rules_test

import (
	"context"
	"testing"

	"github.com/morebec/proxias"
	"github.com/morebec/proxias/rules"
)

func TestRunCommand(t *testing.T) {
	candidates := []proxias.Candidate{{ID: "status"}}
	tests := []struct {
		name, message, want string
	}{
		{"known", "/run status", "status"},
		{"last line wins", "[prefix] hi\n/run status", "status"},
		{"unknown id", "/run ghost", ""},
		{"no command", "how are you?", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := proxias.Request{Messages: []proxias.Message{{Role: proxias.RoleUser, Content: tt.message}}}
			intent, err := rules.RunCommand{}.Classify(context.Background(), request, candidates)
			if err != nil {
				t.Fatal(err)
			}
			got := ""
			if !intent.IsNone() {
				got = intent.WorkflowIDs[0]
			}
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCommandRule(t *testing.T) {
	classifier := rules.New(rules.Command("status", "app_status"))
	request := proxias.Request{Messages: []proxias.Message{{Role: proxias.RoleUser, Content: "/status now"}}}
	intent, _ := classifier.Classify(context.Background(), request, nil)
	if intent.IsNone() || intent.WorkflowIDs[0] != "app_status" {
		t.Errorf("intent = %v", intent)
	}
}
