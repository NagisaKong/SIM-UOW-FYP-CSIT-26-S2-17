# Contributing

## Branching Strategy

- `main` — stable, production-ready code
- `develop` — active development branch
- `feature/xxx` — new features, branched from `develop`

## Pull Request Rules

- All PRs must target `develop`, never directly to `main`
- Merges to `main` happen only from `develop` after review

## Commit Message Format

Use prefixed commit messages:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation changes
- `chore:` — maintenance tasks (deps, config, etc.)
