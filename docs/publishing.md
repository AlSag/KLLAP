# Publishing updates on GitHub

The repository already has an `origin/main`; do not re-run `git init` or replace
the remote. Work on a review branch so the scientific update can be inspected
before merging.

## Prepare the branch

```bash
git switch -c repository-update-2026-10
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

A clear history would use two commits:

```bash
git add README.md CITATION.cff LICENSE THIRD_PARTY_NOTICES.md AI_USAGE.md \
  REPOSITORY_AUDIT.md docs .gitignore .github tests/test_repository_metadata.py
git commit -m "Document current KNe analysis workflow and provenance"

git add \
  generate_kne_lightcurves.py build_color_envelopes.py kllap/generation/backends.py \
  analyze_envelope_population_convergence_fast.py \
  kllap/convergence/colour.py kllap/convergence/multiquantile.py \
  plot_kne_colour_isocontours.py analyze_kne_latitude.py tests
git commit -m "Add multiquantile convergence and parameter-space analyses"
```

Adjust the second list after inspecting `git status`; never use it as a reason
to stage generated products.

## Push and review

```bash
git push -u origin repository-update-2026-10
```

Open a pull request into `main`, confirm the CI result, review the rendered
README and citation metadata, and merge only after the scientific diff has been
checked. Create the `v0.2.0` tag only if that release number and date are
confirmed.

Keep private keys outside the repository and use an SSH agent or Git credential
manager. Large reproducibility products belong in an external data archive,
with checksums and an immutable citation in the repository.
