# Contributing

- Commit messages and pull request titles are [Conventional Commits](https://www.conventionalcommits.org/) (`feat: ...`, `fix(rules): ...`, `chore: ...`). Pull requests are squash-merged, so the title is the changelog line; a CI check enforces it.
- Releases are made by [Release Please](https://github.com/googleapis/release-please): merging its release pull request tags `vX.Y.Z`.
- Before pushing: `gofmt -l .`, `go vet ./...`, `go test -race ./...`.
- Proxias depends on the standard library only, and knows nothing of any host application: what a host needs to pass along travels in `Request.Meta`.
