package schema_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias/schema"
)

func TestObjectSchema_Map(t *testing.T) {
	tests := []struct {
		name string
		when *schema.ObjectSchema
		then map[string]any
	}{
		{
			name: "Canonical case: each kind of property, the required ones listed in order",
			when: schema.Object().
				String("title", "In a few words.").Required().
				Strings("checklist", "The steps.").
				Enum("status", "", "todo", "done").Required().
				Integer("minutes", "How long.").Min(1).Max(1440).
				Bool("done", "Whether it is done.").
				AnyObject("inputs", "By name.").
				Objects("items", "At least one.", schema.Object().
					String("id", "Its ID.").Required()),
			then: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"title": map[string]any{
						"type":        "string",
						"description": "In a few words.",
					},
					"checklist": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "string",
						},
						"description": "The steps.",
					},
					"status": map[string]any{
						"type": "string",
						"enum": []string{
							"todo",
							"done",
						},
					},
					"minutes": map[string]any{
						"type":        "integer",
						"minimum":     1,
						"maximum":     1440,
						"description": "How long.",
					},
					"done": map[string]any{
						"type":        "boolean",
						"description": "Whether it is done.",
					},
					"inputs": map[string]any{
						"type":        "object",
						"description": "By name.",
					},
					"items": map[string]any{
						"type":        "array",
						"description": "At least one.",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"id": map[string]any{
									"type":        "string",
									"description": "Its ID.",
								},
							},
							"required": []string{
								"id",
							},
						},
					},
				},
				"required": []string{
					"title",
					"status",
				},
			},
		},
		{
			name: "GIVEN no property, THEN an object without properties nor required",
			when: schema.Object(),
			then: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.then, tt.when.Map())
		})
	}
}
