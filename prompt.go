package proxias

import "strings"

const (
	promptParagraph promptBlockKind = iota
	promptSection
	promptBulletPoint
)

type promptBlockKind int

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
	kind promptBlockKind
	text string
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
		b.WriteString(block.text)
	}

	return b.String()
}

func (p *Prompt) separator(previous, next promptBlock) string {
	if next.kind == promptBulletPoint && previous.kind != promptParagraph {
		return "\n"
	}
	return "\n\n"
}

func (p *Prompt) add(kind promptBlockKind, text string) *Prompt {
	if text != "" {
		p.blocks = append(p.blocks, promptBlock{kind: kind, text: text})
	}
	return p
}
