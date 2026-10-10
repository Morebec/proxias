package proxias_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/morebec/proxias"
)

func TestPromptString(t *testing.T) {
	tests := []struct {
		name  string
		given func() *proxias.Prompt
		then  string
	}{
		{
			name: "canonical",
			given: func() *proxias.Prompt {
				return proxias.NewPrompt().
					AddParagraph("You decide.").
					AddSection("Rules", "Follow them.").
					AddBulletPoint("Be brief.").
					AddLabeledBulletPoint("restart", "Restarts a service")
			},
			then: `You decide.

## Rules
Follow them.
- Be brief.
- restart: Restarts a service`,
		},
		{
			name:  "GIVEN no block THEN the prompt is empty",
			given: proxias.NewPrompt,
			then:  "",
		},
		{
			name: "GIVEN a section without content THEN it is its title",
			given: func() *proxias.Prompt {
				return proxias.NewPrompt().AddSection("Workflows", "")
			},
			then: "## Workflows",
		},
		{
			name: "GIVEN two sections THEN a blank line separates them",
			given: func() *proxias.Prompt {
				return proxias.NewPrompt().AddSection("A", "a").AddSection("B", "b")
			},
			then: `## A
a

## B
b`,
		},
		{
			name: "GIVEN bullet points after a paragraph THEN a blank line separates them",
			given: func() *proxias.Prompt {
				return proxias.NewPrompt().AddParagraph("Rules:").AddBulletPoint("one").AddBulletPoint("two")
			},
			then: `Rules:

- one
- two`,
		},
		{
			name: "GIVEN a paragraph after bullet points THEN a blank line separates them",
			given: func() *proxias.Prompt {
				return proxias.NewPrompt().AddBulletPoint("one").AddParagraph("Done.")
			},
			then: `- one

Done.`,
		},
		{
			name: "GIVEN a bullet point of several lines THEN its lines are indented under it",
			given: func() *proxias.Prompt {
				return proxias.NewPrompt().AddBulletPoint(`one
two`)
			},
			then: `- one
  two`,
		},
		{
			name: "GIVEN a labeled bullet point without text THEN it is its label",
			given: func() *proxias.Prompt {
				return proxias.NewPrompt().AddLabeledBulletPoint("restart", " ")
			},
			then: "- restart",
		},
		{
			name: "GIVEN blank blocks THEN they are ignored",
			given: func() *proxias.Prompt {
				return proxias.NewPrompt().AddParagraph(" ").AddSection(" ", "x").AddBulletPoint("").AddLabeledBulletPoint("", "x")
			},
			then: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.then, tt.given().String())
		})
	}
}
