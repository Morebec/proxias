package proxias

import (
	"context"
	"errors"
	"fmt"
)

// UserIdentity identifies the user on whose behalf a request is made, as a host defines it (an ID, an email).
// The empty identity means the user is unknown.
type UserIdentity string

// AgentIdentity identifies the agent (the harness or application) that makes a request, as a host defines it.
// The empty identity means the agent is unknown.
type AgentIdentity string

// IdentityProvider tells who a request is from. It is generic so that the same shape serves the user and the
// agent, see UserIdentityProvider and AgentIdentityProvider. An error means the identity could not be
// determined, and fails the request; an unknown identity is the zero value of T, not an error.
type IdentityProvider[T any] interface {
	ProvideIdentity(ctx context.Context, req ChatCompletionRequest) (T, error)
}

// IdentityProviderFunc is an IdentityProvider of a function, for an identity that comes from anywhere: a token,
// a session, a database.
type IdentityProviderFunc[T any] func(ctx context.Context, req ChatCompletionRequest) (T, error)

// ProvideIdentity calls f.
func (f IdentityProviderFunc[T]) ProvideIdentity(ctx context.Context, req ChatCompletionRequest) (T, error) {
	return f(ctx, req)
}

// UserIdentityProvider provides the identity of the user of a request.
type UserIdentityProvider IdentityProvider[UserIdentity]

// AgentIdentityProvider provides the identity of the agent that made a request.
type AgentIdentityProvider IdentityProvider[AgentIdentity]

// HeaderIdentityProvider provides an identity from a header of the HTTP request that carried the request. It
// trusts the header: it is for a host that sits behind something that sets it (a gateway) or that checked it
// already. A request that did not come over HTTP, or that lacks the header, has the zero identity.
type HeaderIdentityProvider[T ~string] struct {
	// HeaderName is the name of the header to read, in any case.
	HeaderName string
}

// NewHeaderIdentityProvider returns a HeaderIdentityProvider that reads the header of the given name.
func NewHeaderIdentityProvider[T ~string](headerName string) *HeaderIdentityProvider[T] {
	return &HeaderIdentityProvider[T]{
		HeaderName: headerName,
	}
}

// DefaultUserIdentityHeader and DefaultAgentIdentityHeader are the headers that NewUserHeaderIdentityProvider
// and NewAgentHeaderIdentityProvider read when given no name: the headers that OpenAI clients send. OpenAI has
// no header for an end user, so the user default is the organization that the client is billed to, which is
// the tenant a proxy usually sees; the agent default is the User-Agent that the OpenAI SDKs and the harnesses
// built on them set to name themselves (for instance "OpenAI/Python 1.40.0").
const (
	DefaultUserIdentityHeader  = "OpenAI-Organization"
	DefaultAgentIdentityHeader = "User-Agent"
)

// NewUserHeaderIdentityProvider returns a HeaderIdentityProvider of UserIdentity that reads the given header,
// or DefaultUserIdentityHeader if headerName is empty.
func NewUserHeaderIdentityProvider(headerName string) *HeaderIdentityProvider[UserIdentity] {
	if headerName == "" {
		headerName = DefaultUserIdentityHeader
	}
	return NewHeaderIdentityProvider[UserIdentity](headerName)
}

// NewAgentHeaderIdentityProvider returns a HeaderIdentityProvider of AgentIdentity that reads the given header,
// or DefaultAgentIdentityHeader if headerName is empty.
func NewAgentHeaderIdentityProvider(headerName string) *HeaderIdentityProvider[AgentIdentity] {
	if headerName == "" {
		headerName = DefaultAgentIdentityHeader
	}
	return NewHeaderIdentityProvider[AgentIdentity](headerName)
}

// ProvideIdentity returns the value of the header, which is empty when the request does not carry it. It never
// fails.
func (h *HeaderIdentityProvider[T]) ProvideIdentity(_ context.Context, req ChatCompletionRequest) (T, error) {
	return T(req.Header.Get(h.HeaderName)), nil
}

// identityKey is the key of an identity of type T in a context: unexported, so that only this package sets it,
// and one per type, so that identities of different types do not collide.
type identityKey[T any] struct{}

// WithIdentity returns a context that carries an identity of any type T, one per type: use it for an identity
// that is a struct of the host's (with a name, an email) rather than a UserIdentity.
// NewIdentityMiddleware sets it for the rest of the chain.
func WithIdentity[T any](ctx context.Context, id T) context.Context {
	return context.WithValue(ctx, identityKey[T]{}, id)
}

// IdentityFrom returns the identity of type T that the context carries, or the zero value of T if it carries
// none.
func IdentityFrom[T any](ctx context.Context) T {
	id, _ := ctx.Value(identityKey[T]{}).(T)
	return id
}

// WithUserIdentity returns a context that carries the identity of the user of a request, as
// WithIdentity[UserIdentity] does.
func WithUserIdentity(ctx context.Context, id UserIdentity) context.Context {
	return WithIdentity(ctx, id)
}

// UserIdentityFrom returns the user identity that the context carries, or the empty identity if it carries none.
func UserIdentityFrom(ctx context.Context) UserIdentity {
	return IdentityFrom[UserIdentity](ctx)
}

// WithAgentIdentity returns a context that carries the identity of the agent that made a request, as
// WithIdentity[AgentIdentity] does.
func WithAgentIdentity(ctx context.Context, id AgentIdentity) context.Context {
	return WithIdentity(ctx, id)
}

// AgentIdentityFrom returns the agent identity that the context carries, or the empty identity if it carries
// none.
func AgentIdentityFrom(ctx context.Context) AgentIdentity {
	return IdentityFrom[AgentIdentity](ctx)
}

// NewIdentityMiddleware returns a Middleware that asks the provider who a request is from, and puts the
// identity in the context for the rest of the chain, to read with IdentityFrom[T]. It never changes or answers
// the request. An error of the provider is returned wrapped with its stage, and the rest of the chain does not
// run. It returns an error if the provider is nil.
func NewIdentityMiddleware[T any](p IdentityProvider[T]) (Middleware, error) {
	if p == nil {
		return nil, errors.New("proxias: NewIdentityMiddleware needs an IdentityProvider")
	}
	return identityMiddleware("provide identity", p), nil
}

// NewUserIdentityMiddleware is NewIdentityMiddleware for the UserIdentity: the rest of the chain reads it with
// UserIdentityFrom. It returns an error if the provider is nil.
func NewUserIdentityMiddleware(p UserIdentityProvider) (Middleware, error) {
	if p == nil {
		return nil, errors.New("proxias: NewUserIdentityMiddleware needs a UserIdentityProvider")
	}
	return identityMiddleware("provide user identity", IdentityProvider[UserIdentity](p)), nil
}

// NewAgentIdentityMiddleware is NewIdentityMiddleware for the AgentIdentity: the rest of the chain reads it
// with AgentIdentityFrom. It returns an error if the provider is nil.
func NewAgentIdentityMiddleware(p AgentIdentityProvider) (Middleware, error) {
	if p == nil {
		return nil, errors.New("proxias: NewAgentIdentityMiddleware needs an AgentIdentityProvider")
	}
	return identityMiddleware("provide agent identity", IdentityProvider[AgentIdentity](p)), nil
}

// identityMiddleware is the step that the identity middleware share: ask the provider, put the answer in the
// context, and pass on.
func identityMiddleware[T any](stage string, p IdentityProvider[T]) Middleware {
	return func(next ProcessFunc) ProcessFunc {
		return func(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
			id, err := p.ProvideIdentity(ctx, req)
			if err != nil {
				return ChatCompletionResponse{}, fmt.Errorf("%s: %w", stage, err)
			}
			return next(WithIdentity(ctx, id), req)
		}
	}
}
