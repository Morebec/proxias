# Contributing

- Commit messages and pull request titles are [Conventional Commits](https://www.conventionalcommits.org/) (`feat: ...`, `fix(rules): ...`, `chore: ...`). Pull requests are squash-merged, so the title is the changelog line; a CI check enforces it.
- Releases are made by [Release Please](https://github.com/googleapis/release-please): merging its release pull request tags `vX.Y.Z`.
- Tasks are run with [Task](https://taskfile.dev) (`task --list`). Before pushing, run `task check`: it does what CI does (gofmt, golangci-lint, `go test -race ./...` and a benchmark smoke run). `task lint`, `task test` and `task fmt` run on their own; `task deadcode` lists what no test or example reaches.
- Linting is [golangci-lint](https://golangci-lint.run) v2, configured in `.golangci.yml`. `task` runs the version pinned in `Taskfile.yml`, which must stay in step with the one of `.github/workflows/ci.yml`.
- Proxias keeps its dependencies to a minimum (the official Anthropic SDK for `adapters/anthropic`; its tests use testify), and knows nothing of any host application: what a host needs to pass along travels in `CompletionRequest.Request`.
