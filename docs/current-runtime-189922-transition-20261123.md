# Current runtime 189922 transition ledger (2026-11-23)

## Boundary

This ledger records the transition to parser runtime
`18992265d9cc6337aac12cab6c88307cbe5a1b855c39844e7681c6f7df2885b8`.
It does not authorize model training. The immutable source freeze, historical task
databases, current task databases, reviewed instruction database, and external
evaluation sources remain separate artifacts.

The reusable-task migration code was frozen at commit `90b473b`. It permits a task
row to move into a new current-runtime database only when the historical database
has a passed v2 audit, the admission-delta receipt proves that document unchanged,
and the document remains eligible under runtime 189922. Validation and test reuse
also require their exact forced split.

## Current source evidence

Fresh admission, semantic-audit, and eligibility evidence covers the complete
frozen inventory.

| Split | Source documents | Eligible | Source rejects | Semantic audit findings |
| --- | ---: | ---: | ---: | ---: |
| train | 25,747 | 25,711 | 36 | 0 |
| validation | 3,219 | 3,216 | 3 | 0 |
| test | 3,207 | 3,199 | 8 | 0 |
| reserved held-out | 369 | 367 | 2 | 0 |

The rejected-source review found bounded transport failures and comment-only or
reference-only inputs, not an unexplained sidecar crash. These rejects are not
relabeled as incorrect semantic records.

## Current evaluation task databases

The validation and test databases were created on BossCrafts MSI under
`C:\BCNLT-runs\current-189922`. Each database completed ordinary packing after
verified reuse, passed the packer's built-in full audit, and then passed a new
independent v2 full-database audit over the finished file.

| Split | Documents | Examples | Packing rejects | Sequence tokens | Database identity | Content SHA-256 | Independent manifest | Audit file SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| validation | 3,216 | 3,824,845 | 22,814 | 3,894,585,405 | `52f71e8f01b36796aa7973ce27770bfd5fc9a8930735c7b233191a44321d917b` | `540f0be52a3a1911c7cc6b584aa11fa5551aa654b24ef1b25534fd3f64cf99b3` | `a930aaa8f584de8442afd8473689538417a44330fa1ba3152e5b195368e69443` | `ddaf322965ca414e164176900db21ea0abaf4aa54f509f6cb65a072711901b8f` |
| test | 3,199 | 3,632,895 | 20,324 | 3,693,345,598 | `67ee2aee9a9ef946f48516b4fc4f1a7f4658ecd3982ad6f58b4c78d1ef5a238b` | `f57ad9eaed43d2e2ed3468459b2ba61f5afc54d937ba84decbf7c68017bcad8a` | `a96050bee297679e18948481fe393ef8598187e6dc0e8853ccb2436e7b56fdde` | `6cc029368891e9c616a7f3e80033cae5e4a2b21abd633b94a73b0c571001d824` |

The reserved held-out database is separately frozen at 367 documents and 468,727
examples. Its database identity is
`07346d7a47a3a5a0bc4d5bcb8f199e497058b7f1747ca2e0530080136aeb5ba8`,
and its independent v2 audit passed. It is test-only and is not part of the train
inventory.

## Reviewed instruction evidence

The reviewed instruction source was rebuilt rather than relabeled. The current
database contains 57,943 examples split as 46,354 train, 5,731 validation, and
5,858 test. Its logical content SHA-256 remains
`bcd57661933cb186e7df19d45556bea9d0342e7d81a310e49cc72f0a5beec346`,
and its independent v2 audit passed. Only its train rows may enter a future mixed
curriculum.

## Natural calibration preflight

The complete current validation and test semantic audits produced a disjoint
calibration preflight at
`C:\BCNLT-runs\calibration-current-189922-witness2\audited-v1` on PATISBORG.
The manifest identity is
`cf1570541d910339fc73c1c60e1f9e05a40d8e4365ca675c9938f9b3fb75558e`,
and the manifest file SHA-256 is
`92ba7e4019eb4d33a128e92eb4da0df7eedea0a269073cbee64e05597a741b49`.

It contains 140 calibration and 140 held-out observations, ten per language in
each split. All 280 are qualified keeps. The artifact correctly remains blocked
by one missing-reject condition for every language in both splits. Zero observed
errors are positive coverage evidence, but they are not a two-class calibration
dataset.

## Pinned external hybrid probe

McEval revision `c4767fec950de7decd487a40a4dab2795df4a094` was acquired from
Hugging Face as evaluation-only CC-BY-SA-4.0 data. The acquisition tool processes
every generation case for all 14 BCNLT languages and does not add that license to
the training-source allowlist.

- Source root: `C:\BCNLT-runs\external-eval-sources\mceval-c4767fec950d`
- Source manifest SHA-256:
  `d3d3301e073d9d4cdc4f7cb7fe2554e72e4086ebebd282058df18c4057b0be13`
- Probe root:
  `C:\BCNLT-runs\external-eval-probes\mceval-c4767fec-current-189922-v3`
- Probe manifest identity:
  `4dab5a9c088f9ff9ec75efde5cdba98ef34f761fceb50460266dad00815e7509`
- Probe manifest file SHA-256:
  `9bf4edf5db3d03461608ff0331d9012a5d882e586a93e72eb1e83a1032adf279`

The probe processed all 706 documents. It independently checked 106,261 semantic
records from 698 documents and found zero deterministic or raw-AST audit findings.
Six Bash and two C++ documents contained Tree-sitter `ERROR` or `MISSING`
diagnostics. Direct inspection
localized those diagnostics to grammar-edge constructs including Bash substring or
array syntax and C++ `va_arg` type arguments. All eight entered bounded `raw_code`
fallback, so the observed hybrid coverage is 706/706.

This probe is diagnostic evidence, not a qualified calibration-label source and
not training data.

## Verification of the code change

- Focused acquisition, category-source, corpus, curriculum, inventory, and gate
  tests: passed.
- Ruff: passed.
- Mypy: 93 source files passed.
- Full non-Docker regression in the project virtual environment: 724 passed,
  21 Docker-only tests deselected, 210 subtests passed in 701.39 seconds.

An attempted full run under the system Anaconda environment aborted inside NumPy
BLAS initialization before test collection. Direct NumPy and Torch imports passed,
and the complete project-virtual-environment run above is the valid regression
result. The aborted system-environment run is not counted as test evidence.

## Remaining gate order

1. Finish the historical train task pack and its independent v2 audit.
2. Migrate only proved-equivalent train rows into a new runtime-189922 database,
   pack every remaining document, and independently audit the finished database.
3. Run the combined loss-mask audit over the final train, validation, and test
   databases.
4. Build the train-only mixed curriculum while preserving every held-out split.
5. Resolve the uncertainty-policy decision without fabricating natural negatives.
   The current safe alternative is to keep adaptive learning disabled and use the
   proven hybrid fallback until a genuine two-class calibration corpus exists.
6. Run final readiness gates before starting any model-weight training.

Training remains held.
