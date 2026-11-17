# Reproducible development environment

This isolated candidate requires 64-bit CPython 3.13 on Windows. Its exact
BCNLT tokenizer, native tokenizer runtime, Tree-sitter bindings, and 14 grammar
libraries are stored in Git LFS and verified by their existing content hashes
before use.

From a fresh clone:

```powershell
git lfs install
git lfs pull
git lfs fsck
uv sync --dev
uv run pytest -q
uv run mypy bosscrafts_model
```

The lock defines the host Python packages. The project-owned tokenizer and
parser packages are deliberately not installed from PyPI; the runtime loads
the content-addressed copies under `vendor/` and rejects any hash mismatch.
The locked `torch==2.11.0` constraint also accepts the locally verified
`2.11.0+cu130` CUDA build. GPU hosts should configure a trusted official
PyTorch CUDA wheel index before syncing if they require that local build.

Generated corpora, admission databases, review evidence, checkpoints, and
model exports remain outside Git. Their immutable identities belong in
lineage manifests and acceptance reports, not in the source repository.

The Git tree is the review boundary. Pull requests and automated reviewers
such as CodeRabbit can inspect all source, tests, manifests, and lock files
without access to this workstation or its OneDrive paths. Native binaries are
Git LFS objects because they are executable dependencies rather than useful
line-by-line review targets; their hashes and provenance remain reviewable in
the ordinary Git files beside them.
