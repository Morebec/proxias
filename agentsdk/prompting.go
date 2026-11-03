package agentsdk

import (
	"strings"
)

// unattendedNote is what the premade prompts add when the agent is invoked unattended.
//
//nolint:unused // not used since the invocation modes went: kept for when they come back
const unattendedNote = `

Nobody is reading your replies: you work unattended, woken by what happens. Act when something needs doing; answer only when someone should hear it, and say who. Silence is fine: when nothing needs doing, end with an empty answer.`

const (
	promptLine promptBlockKind = iota
	promptParagraph
	promptSection
	promptBulletPoint
)

type promptBlockKind int

// NewAgentPrompt returns a prompt describing is who the agent is, in words,
// for a strategy that tells a model: its source, then its name, role, goals
// and biases, those it has. Every premade strategy tells its models this, so that they are compared on the same
// context.
func NewAgentPrompt(a Agent) string {
	p := NewPrompt()
	name := "(" + a.AsSource().String() + ")"
	if a.Name != "" {
		name = a.Name + " " + name
	}

	p.AddLine("You are " + name + ".")
	if a.Role != "" {
		p.AddLine("Role: " + a.Role)
	}

	if len(a.Goals) > 0 {
		p.AddLine("Goals:")
		for _, goal := range a.Goals {
			p.AddBulletPoint(goal)
		}
	}

	if len(a.Biases) > 0 {
		p.AddLine("Behavioural Guidelines:")
		for _, bias := range a.Biases {
			p.AddBulletPoint(string(bias))
		}
	}

	return p.String()
}

// Prompt builds the text of a prompt for an LLM from paragraphs, titled sections and bullet points, so that
// the components that talk to a model do not assemble strings by hand. The text is plain Markdown: a section
// is a "## " heading, a bullet point a "- " line.
//
// Blocks come out in the order they were added. Consecutive bullet points, and bullet points that follow a
// section, are on consecutive lines (a bullet point list belongs to the section above it); everything else is
// separated by a blank line. A Prompt is not safe for concurrent use. Use NewPrompt; its methods return the
// Prompt, to chain them.
type Prompt struct {
	blocks []promptBlock
}

type promptBlock struct {
	kind    promptBlockKind
	content string
}

// NewPrompt makes an empty Prompt.
func NewPrompt() *Prompt {
	return &Prompt{}
}

// AddParagraph adds text as is. A blank text is ignored.
func (p *Prompt) AddParagraph(text string) *Prompt {
	return p.add(promptParagraph, strings.TrimSpace(text))
}

// AddSection adds a section: a title, and its content, if any. Bullet points added right after it are part
// of the section. A blank title is ignored.
func (p *Prompt) AddSection(title, content string) *Prompt {
	title, content = strings.TrimSpace(title), strings.TrimSpace(content)
	if title == "" {
		return p
	}

	text := "## " + title
	if content != "" {
		text += "\n" + content
	}
	return p.add(promptSection, text)
}

// AddLine adds a line, right under the block before it unless that is a paragraph. A blank line is ignored.
func (p *Prompt) AddLine(line string) *Prompt {
	line = strings.TrimSpace(line)
	if line == "" {
		return p
	}
	return p.add(promptLine, line)
}

// AddBulletPoint adds a bullet point. A blank text is ignored. The lines of a text that has several are
// indented under its bullet.
func (p *Prompt) AddBulletPoint(text string) *Prompt {
	text = strings.TrimSpace(text)
	if text == "" {
		return p
	}
	return p.add(promptBulletPoint, "- "+strings.ReplaceAll(text, "\n", "\n  "))
}

// AddLabeledBulletPoint adds a bullet point of the form "- label: text", to list things that have a name
// (an identifier, a term) along with what they are. A blank label is ignored, and a blank text leaves only
// the label.
func (p *Prompt) AddLabeledBulletPoint(label, text string) *Prompt {
	label, text = strings.TrimSpace(label), strings.TrimSpace(text)
	if label == "" {
		return p
	}
	if text == "" {
		return p.AddBulletPoint(label)
	}
	return p.AddBulletPoint(label + ": " + text)
}

// String renders the prompt.
func (p *Prompt) String() string {
	var b strings.Builder

	for i, block := range p.blocks {
		if i > 0 {
			b.WriteString(p.separator(p.blocks[i-1], block))
		}
		b.WriteString(block.content)
	}

	return b.String()
}

func (p *Prompt) separator(previous, next promptBlock) string {
	if (next.kind == promptBulletPoint || next.kind == promptLine) &&
		previous.kind != promptParagraph {
		return "\n"
	}
	return "\n\n"
}

func (p *Prompt) add(kind promptBlockKind, text string) *Prompt {
	if text != "" {
		p.blocks = append(p.blocks, promptBlock{kind: kind, content: text})
	}
	return p
}
