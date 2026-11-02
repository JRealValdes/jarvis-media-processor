---
name: github-feature-flow
description: >-
  Create and maintain GitHub issues, feature branches named feature/X-slug,
  and PRs into main with titles ending in (#X) for jarvis-media-processor.
  Use when starting work from main, solving or continuing an existing GitHub
  issue, creating or updating an issue for a feature, opening a feature
  branch, linking a PR to an issue, or when the user asks to start the
  issue/branch/PR workflow.
---

# GitHub feature flow

Issue + feature branch + PR into `main`. All GitHub text in **English**.

Repo: `JRealValdes/jarvis-media-processor`.

## Start from `main` — new work (propose first)

If the user asks for a **new** change while on `main` (no issue number), **propose**, then wait for confirmation:

1. Create a GitHub issue in this repo
2. Create branch `feature/<N>-<slug>` from up-to-date `main` and develop there

Do not create the issue or branch until they agree (unless they already ordered the full flow).

## Work on an existing issue `#N` (from `main` or elsewhere)

If the user asks to solve, implement, or continue issue `#N`:

1. **Read the issue:** `gh issue view <N>` (optionally `--comments`). Treat the issue body/checklist as the source of requirements.
2. **Use the development branch:**
   - Look for an existing local or remote branch matching `feature/<N>-*` (or the branch linked under the issue’s Development section).
   - If it exists: check it out and pull latest (`git fetch` / `git pull`).
   - If it does not exist: create `feature/<N>-<slug>` from up-to-date `main` (slug from the issue title). Prefer `gh issue develop <N> --name feature/<N>-<slug> --checkout` when available; otherwise `git checkout -b` + `git push -u`.
3. **Develop on that branch** only — do not implement `#N` on `main`. Update the issue body when new scope materially helps tracking.

## 1. Create the issue

```bash
gh issue create --title "<English title>" --body-file <path-or-heredoc>
```

Body: short context, goal, checklist. English only.

Record the issue number from the URL or `gh issue list`.

## 2. Create the branch

Slug: lowercase kebab-case from the issue title; drop punctuation; keep it short.

```bash
git fetch origin
git checkout main
git pull origin main
git checkout -b feature/<N>-<slug>
git push -u origin HEAD
```

Prefer linking development in GitHub when available:

```bash
gh issue develop <N> --name feature/<N>-<slug> --checkout
```

If `gh issue develop` is unavailable, the branch name + PR link is enough.

## 3. Develop

- Commit with Conventional Commits; only when the user asks to commit (or confirms a proposed commit — see incremental-commits rules).
- Verify with unit tests before proposing commits when the change has non-trivial logic.
- If scope grows and it **materially** helps tracking, update the issue body (checklist / notes) via `gh issue edit <N> --body-file ...` **without asking**. Skip trivial or redundant edits.

## 4. When the task is done — propose close-out

When the issue work is **complete**, verified, and **fully committed** on `feature/<N>-*`, **proactively propose** (do not wait to be asked):

1. Push the branch  
2. Open a PR into `main`  
3. Merge the PR  
4. Clean up remote + local feature branches, then land on updated `main`

Wait for user confirmation before executing. The user may approve the whole chain in one reply.

## 5. Push and open a PR (after confirmation)

```bash
git push -u origin HEAD
gh pr create --title "<English summary> (#<N>)" --body-file <file>
```

**Title rule:** the title must end with a space and `(#N)` — e.g. `Add FFmpeg transcode pipeline (#8)`.

**Body (English), include:**

```markdown
## Summary
- …

Closes #<N>.

## Test plan
- [ ] …
```

`Closes #<N>` associates the PR with the issue and closes it on merge. Use `Refs #<N>` instead if the PR is partial and must not close the issue.

## 6. Merge and branch cleanup (after confirmation)

```bash
gh pr merge <PR> --merge   # or --squash if the user prefers
git fetch --prune
git checkout main
git pull origin main
git push origin --delete feature/<N>-<slug>   # if remote branch still exists
git branch -d feature/<N>-<slug>              # -D only if needed after squash/rebase
```

End on `main` with a clean working tree matching `origin/main`.

## Checklist

```
- [ ] Issue exists (#N), English; read before coding when working an existing issue
- [ ] On feature/<N>-* (checked out or created), not on main
- [ ] Work landed on that branch
- [ ] Issue body updated only when useful
- [ ] When done + committed: propose push → PR → merge → branch cleanup
- [ ] PR title ends with (#N); body links Closes/Refs #N
- [ ] After confirmed merge: remote/local feature branches removed; on main and pulled
```
