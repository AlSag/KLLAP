# Publishing on GitHub

Perform these steps only after resolving the placeholders listed in
`REPOSITORY_AUDIT.md`.

```bash
git init
git branch -M main
git add .
git status --short
git diff --cached --check
git commit -m "Initial public KNe-LSST analysis pipeline"
```

Create an empty repository on GitHub, then use either an SSH or HTTPS remote:

```bash
git remote add origin git@github.com:OWNER/REPOSITORY.git
git push -u origin main
```

Do not place a private key in this directory. SSH private keys belong in the
user's SSH configuration outside the repository, with restrictive filesystem
permissions and an SSH agent. Before the first push, inspect the complete
staged file list and run a secret scanner if one is available.

Recommended repository settings:

- enable branch protection for `main`;
- require pull-request review and passing tests;
- enable GitHub secret scanning and dependency alerts;
- create a tagged release for every scientific result set;
- archive large inputs/outputs separately and publish checksums or DOI-backed
  data records rather than committing them to Git.
