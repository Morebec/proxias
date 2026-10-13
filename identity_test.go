package proxias_test

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/morebec/proxias"
)

// mockUserIdentityProvider provides the given identity, or fails with err.
func mockUserIdentityProvider(id proxias.UserIdentity, err error) proxias.UserIdentityProvider {
	return mockIdentityProvider[proxias.UserIdentity]{id: id, err: err}
}

// mockAgentIdentityProvider provides the given identity, or fails with err.
func mockAgentIdentityProvider(id proxias.AgentIdentity, err error) proxias.AgentIdentityProvider {
	return mockIdentityProvider[proxias.AgentIdentity]{id: id, err: err}
}

type mockIdentityProvider[T any] struct {
	id  T
	err error
}

func (m mockIdentityProvider[T]) ProvideIdentity(context.Context, proxias.ChatCompletionRequest) (T, error) {
	return m.id, m.err
}

func TestHeaderIdentityProviderProvideIdentity(t *testing.T) {
	tests := []struct {
		name  string
		given string
		when  http.Header
		then  proxias.UserIdentity
	}{
		{
			name:  "canonical",
			given: "X-User-Id",
			when:  http.Header{"X-User-Id": {"ada"}},
			then:  "ada",
		},
		{
			name:  "GIVEN a header name in another case WHEN the header is present THEN it is found",
			given: "x-user-id",
			when:  http.Header{"X-User-Id": {"ada"}},
			then:  "ada",
		},
		{
			name:  "GIVEN a header with several values WHEN provided THEN it is the first",
			given: "X-User-Id",
			when:  http.Header{"X-User-Id": {"ada", "bob"}},
			then:  "ada",
		},
		{
			name:  "WHEN the header is absent THEN the identity is empty",
			given: "X-User-Id",
			when:  http.Header{"X-Other": {"ada"}},
			then:  "",
		},
		{
			name:  "WHEN the request has no headers THEN the identity is empty",
			given: "X-User-Id",
			when:  nil,
			then:  "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Given
			provider := proxias.NewHeaderIdentityProvider[proxias.UserIdentity](tt.given)

			// When
			got, err := provider.ProvideIdentity(t.Context(), proxias.ChatCompletionRequest{Header: tt.when})

			// Then
			require.NoError(t, err)
			assert.Equal(t, tt.then, got)
		})
	}
}

func TestHeaderIdentityProviderIsAnIdentityProvider(t *testing.T) {
	// Given a header provider of each identity type
	var user proxias.UserIdentityProvider = proxias.NewHeaderIdentityProvider[proxias.UserIdentity]("X-User-Id")
	var agent proxias.AgentIdentityProvider = proxias.NewHeaderIdentityProvider[proxias.AgentIdentity]("X-Agent-Id")
	req := proxias.ChatCompletionRequest{Header: http.Header{"X-User-Id": {"ada"}, "X-Agent-Id": {"cli"}}}

	// When they provide
	u, err := user.ProvideIdentity(t.Context(), req)
	require.NoError(t, err)
	a, err := agent.ProvideIdentity(t.Context(), req)
	require.NoError(t, err)

	// Then each reads its own header
	assert.Equal(t, proxias.UserIdentity("ada"), u)
	assert.Equal(t, proxias.AgentIdentity("cli"), a)
}

func TestIdentityFromWithoutIdentity(t *testing.T) {
	// Given a context that carries no identity, When read, Then both are empty
	assert.Empty(t, proxias.UserIdentityFrom(t.Context()))
	assert.Empty(t, proxias.AgentIdentityFrom(t.Context()))
}

func TestWithIdentityKeepsTheIdentitiesApart(t *testing.T) {
	// Given a context with a user identity and an agent identity
	ctx := proxias.WithAgentIdentity(proxias.WithUserIdentity(t.Context(), "ada"), "cli")

	// When read, Then each is its own
	assert.Equal(t, proxias.UserIdentity("ada"), proxias.UserIdentityFrom(ctx))
	assert.Equal(t, proxias.AgentIdentity("cli"), proxias.AgentIdentityFrom(ctx))
}

func TestNewIdentityMiddlewareRejectsANilProvider(t *testing.T) {
	// Given/When a nil provider
	_, userErr := proxias.NewUserIdentityMiddleware(nil)
	_, agentErr := proxias.NewAgentIdentityMiddleware(nil)

	// Then each constructor returns an error
	assert.ErrorContains(t, userErr, "NewUserIdentityMiddleware needs a UserIdentityProvider")
	assert.ErrorContains(t, agentErr, "NewAgentIdentityMiddleware needs an AgentIdentityProvider")
}

func TestIdentityMiddlewareProvidesTheIdentities(t *testing.T) {
	// Given the user and agent middleware reading the headers, and a step that reads the context and request
	user, err := proxias.NewUserIdentityMiddleware(proxias.NewUserHeaderIdentityProvider("X-User-Id"))
	require.NoError(t, err)
	agent, err := proxias.NewAgentIdentityMiddleware(proxias.NewAgentHeaderIdentityProvider("X-Agent-Id"))
	require.NoError(t, err)
	var gotUser proxias.UserIdentity
	var gotAgent proxias.AgentIdentity
	var reached proxias.ChatCompletionRequest
	reader := func(proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, req proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			gotUser, gotAgent, reached = proxias.UserIdentityFrom(ctx), proxias.AgentIdentityFrom(ctx), req
			return someResponse, nil
		}
	}
	controlPlane, err := proxias.NewControlPlane(user, agent, reader)
	require.NoError(t, err)
	req := someRequest
	req.Header = http.Header{"X-User-Id": {"ada"}, "X-Agent-Id": {"cli"}}

	// When a request is processed
	resp, err := controlPlane.Process(t.Context(), req)

	// Then the rest of the chain sees both identities and the request as it was
	require.NoError(t, err)
	assert.Equal(t, someResponse, resp)
	assert.Equal(t, proxias.UserIdentity("ada"), gotUser)
	assert.Equal(t, proxias.AgentIdentity("cli"), gotAgent)
	assert.Equal(t, req, reached)
}

func TestIdentityMiddlewareFailure(t *testing.T) {
	cause := errors.New("boom")
	tests := []struct {
		name  string
		given func() (proxias.Middleware, error)
		then  string
	}{
		{
			name: "GIVEN a failing user provider",
			given: func() (proxias.Middleware, error) {
				return proxias.NewUserIdentityMiddleware(mockUserIdentityProvider("", cause))
			},
			then: "proxias: provide user identity: boom",
		},
		{
			name: "GIVEN a failing agent provider",
			given: func() (proxias.Middleware, error) {
				return proxias.NewAgentIdentityMiddleware(mockAgentIdentityProvider("", cause))
			},
			then: "proxias: provide agent identity: boom",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Given a chain whose next step must not run
			middleware, err := tt.given()
			require.NoError(t, err)
			var order []string
			controlPlane, err := proxias.NewControlPlane(middleware, recordingStep(&order, "never"), passThrough)
			require.NoError(t, err)

			// When a request is processed
			_, err = controlPlane.Process(t.Context(), someRequest)

			// Then the cause is found, the prefix is there once, and nothing after ran
			assert.ErrorIs(t, err, cause)
			assert.EqualError(t, err, tt.then)
			assert.Empty(t, order)
		})
	}
}

func TestNewUserHeaderIdentityProviderDefaultsToTheOpenAIHeader(t *testing.T) {
	// Given a provider made without a header name
	provider := proxias.NewUserHeaderIdentityProvider("")
	req := proxias.ChatCompletionRequest{Header: http.Header{"Openai-Organization": {"org-1"}}}

	// When
	got, err := provider.ProvideIdentity(t.Context(), req)

	// Then it reads the default header
	require.NoError(t, err)
	assert.Equal(t, "OpenAI-Organization", proxias.DefaultUserIdentityHeader)
	assert.Equal(t, proxias.UserIdentity("org-1"), got)
}

func TestNewAgentHeaderIdentityProviderDefaultsToTheUserAgent(t *testing.T) {
	// Given a provider made without a header name, and a request from an OpenAI SDK
	provider := proxias.NewAgentHeaderIdentityProvider("")
	req := proxias.ChatCompletionRequest{Header: http.Header{"User-Agent": {"OpenAI/Python 1.40.0"}}}

	// When
	got, err := provider.ProvideIdentity(t.Context(), req)

	// Then it reads the default header
	require.NoError(t, err)
	assert.Equal(t, "User-Agent", proxias.DefaultAgentIdentityHeader)
	assert.Equal(t, proxias.AgentIdentity("OpenAI/Python 1.40.0"), got)
}

func TestHeaderIdentityProviderConstructorsKeepAGivenName(t *testing.T) {
	// Given providers made with a name, and a request that has it and the defaults
	req := proxias.ChatCompletionRequest{Header: http.Header{
		"X-User-Id": {"ada"}, "X-Agent-Id": {"cli"}, "Openai-Organization": {"org-1"}, "User-Agent": {"sdk"},
	}}

	// When
	user, err := proxias.NewUserHeaderIdentityProvider("X-User-Id").ProvideIdentity(t.Context(), req)
	require.NoError(t, err)
	agent, err := proxias.NewAgentHeaderIdentityProvider("X-Agent-Id").ProvideIdentity(t.Context(), req)
	require.NoError(t, err)

	// Then the name wins over the default
	assert.Equal(t, proxias.UserIdentity("ada"), user)
	assert.Equal(t, proxias.AgentIdentity("cli"), agent)
}

// account is an identity of the host's, richer than a UserIdentity.
type account struct {
	ID       string
	FullName string
	Email    string
}

func TestNewIdentityMiddlewareCarriesAnIdentityOfAnyType(t *testing.T) {
	// Given a provider of a struct identity, and a step that reads it and a user identity
	provider := proxias.IdentityProviderFunc[account](func(context.Context, proxias.ChatCompletionRequest) (account, error) {
		return account{ID: "1", FullName: "Ada Lovelace", Email: "ada@example.com"}, nil
	})
	middleware, err := proxias.NewIdentityMiddleware(provider)
	require.NoError(t, err)
	var got account
	var user proxias.UserIdentity
	reader := func(proxias.ProcessFunc) proxias.ProcessFunc {
		return func(ctx context.Context, _ proxias.ChatCompletionRequest) (proxias.ChatCompletionResponse, error) {
			got, user = proxias.IdentityFrom[account](ctx), proxias.UserIdentityFrom(ctx)
			return someResponse, nil
		}
	}
	controlPlane, err := proxias.NewControlPlane(middleware, reader)
	require.NoError(t, err)

	// When a request is processed
	_, err = controlPlane.Process(t.Context(), someRequest)

	// Then the struct is in the context, and a different type is not
	require.NoError(t, err)
	assert.Equal(t, account{ID: "1", FullName: "Ada Lovelace", Email: "ada@example.com"}, got)
	assert.Empty(t, user)
}

func TestNewIdentityMiddlewareFailure(t *testing.T) {
	// Given a nil provider, and a failing one
	_, nilErr := proxias.NewIdentityMiddleware[account](nil)
	cause := errors.New("boom")
	middleware, err := proxias.NewIdentityMiddleware(proxias.IdentityProviderFunc[account](
		func(context.Context, proxias.ChatCompletionRequest) (account, error) { return account{}, cause }))
	require.NoError(t, err)
	controlPlane, err := proxias.NewControlPlane(middleware, passThrough)
	require.NoError(t, err)

	// When a request is processed
	_, procErr := controlPlane.Process(t.Context(), someRequest)

	// Then the constructor refuses the nil, and the failure is wrapped once
	assert.ErrorContains(t, nilErr, "NewIdentityMiddleware needs an IdentityProvider")
	assert.ErrorIs(t, procErr, cause)
	assert.EqualError(t, procErr, "proxias: provide identity: boom")
}
