# Publishing updates on GitHub

The repository already has an `origin/main`; do not re-run `git init` or replace
the remote. Work on a review branch so the scientific update can be inspected
before merging.

## Prepare the branch

```bash
git switch -c review/photometry-population
git status --short
```

Copy the reviewed update files into the repository, then remove or leave
unstaged local products that are intentionally ignored.

## Validate before staging

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
python - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path('CITATION.cff').read_text())
print('CITATION.cff: valid YAML')
PY
```

Inspect the exact change set:

```bash
git status --short
git diff --check
git diff --stat
git diff -- README.md CITATION.cff THIRD_PARTY_NOTICES.md AI_USAGE.md
```

Check that no generated run, model weight, database, archive, secret, or local
absolute path is staged:

```bash
git add --dry-run .
git grep -nE '/home/|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|ssh_key' -- . ':!docs/publishing.md'
```

## Commit in reviewable units

Stage only the files shown by the reviewed diff. For an architecture update:

```bash
git diff --check
python -m compileall -q .
python -m unittest discover -s tests -v
git add -A
git diff --cached --stat
git commit -m "Describe the reviewed change"
```

Always inspect `git status` before staging; generated products must remain
excluded.

## Push and review

```bash
git push -u origin review/photometry-population
```

Open a pull request into `main`, confirm the CI result, review the rendered
README and citation metadata, and merge only after the scientific diff has been
checked. Create the `v0.2.0` tag only if that release number and date are
confirmed.

Keep private keys outside the repository and use an SSH agent or Git credential
manager. Large reproducibility products belong in an external data archive,
with checksums and an immutable citation in the repository.

## Apply the supplied patch without changing a dirty local checkout

From an existing authenticated clone, fetch the current base and create a
separate worktree. Replace `/path/to/KLLAP_update.patch` with the downloaded
patch path. This preserves uncommitted work in the original checkout.

```bash
git fetch origin
git worktree add -b review/photometry-population ../KLLAP-review origin/main
cd ../KLLAP-review
git am --3way /path/to/KLLAP_update.patch
git push -u origin review/photometry-population
```

If the review branch already exists, choose another branch/worktree name.
Review the scientific changes and migration notes before merging into `main`.
