# BCNLT Native Extended Core Phase — 2026-11-07

## Implemented

- Append-only native token allocation after the exact BCNLT v1 vocabulary.
- Stable token names and IDs across evidence refreshes.
- Decode-only tombstones for revoked meanings; IDs and model rows are never reused.
- Native semantic-stream identity tokens with canonical metadata agreement checks.
- Canonical, hash-validated nested tokenizer artifact save/reload.
- Deterministic embedding and LM-head row growth that preserves all base, v1, and
  earlier Extended Core rows exactly.
- Chained vocabulary growth for later meanings and corrected revisions.

## Verification

- Full main checkout: `.venv\Scripts\python.exe -m pytest -q` → **480 passed**.
- Native vocabulary and embedding focused suite → **19 passed**.
- `python -m compileall -q bcnlt integrations tests` passed.
- `git diff --check` passed.

## Boundary

No model weights, preserved token inventories, training corpora, run artifacts,
commits, or remotes were changed. No real native artifact was fabricated because no
scientifically validated learned-definition ledger was found in the repository or
under `D:\BCNLT-runs`. The next admissible step is to freeze real validated Extended
Core evidence, build its native artifact, and run held-out tokenizer/model gates
before bounded training.
