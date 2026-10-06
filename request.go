package proxias

// Role says who a message comes from.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool" // a tool result: the host continues its own tool loop
)

// Message is one message of a conversation.
type Message struct {
	Role    Role
	Content string
}

// Request is what is intercepted: the conversation, the message to answer last. Meta is the host's own
// data (who asks, the raw request body...), passed untouched to the IntentClassifier and the Fallback;
// the Interceptor never reads it.
type Request struct {
	Messages []Message
	Meta     map[string]any
}

// Last returns the last message of the conversation.
func (r Request) Last() (Message, bool) {
	if len(r.Messages) == 0 {
		return Message{}, false
	}
	return r.Messages[len(r.Messages)-1], true
}
