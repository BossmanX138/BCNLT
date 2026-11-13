# V12 checkpoint recovery and diagnostic

Date: 2026-11-13

## Purpose

Recover the last V12 weights without modifying their source, verify the copy,
and determine whether the incomplete checkpoint is useful enough to resume or
promote. This is diagnostic evidence only. It is not a promotion receipt.

## Preserved source and copies

- Original source:
  `\\BOSSCRAFTS-MSI\R$\BossCrafts-Engineering-0.5B\artifacts\checkpoints\bosscrafts-engineering-v12-codex-replay\checkpoint.pt`
- Durable recovery copy:
  `D:\BCNLT-runs\recovered-checkpoints\bosscrafts-engineering-v12-codex-replay\checkpoint.pt`
- SSD diagnostic copy:
  `C:\Users\bossm\Documents\Codex\BCNLT\.codex_staging_v12_recovered\checkpoint.pt`
- Size: 3,240,810,635 bytes
- SHA-256 for the source and both verified copies:
  `9f745318c43a292ac4d9e0619e30fc29666c15ca92423df3ecc9840111b92935`

The source was not modified. The D-drive copy was compared byte-for-byte by
SHA-256 with the MSI source. The SSD copy was then compared with that same
verified digest.

## Checkpoint state

- Durable optimizer step: 17,200 / 17,922
- Remaining optimizer steps: 722
- Epoch index: 0 / 1
- Training complete: false
- Binding SHA-256:
  `6bdcf5df4e5f16b79ee8de6eea9b86194c1e9909ef8e8545e28d5847ab031268`
- Tokenizer SHA-256:
  `e1930449f3efe99bbb3f63730f75f71788d4ff21928bcae4d3ee2a520d30d407`
- Model fingerprint:
  `383bc3acb9bea01ed9a40fb6293a5ac81ee7ffb27ffbc4fdcfeafd96358b3bca`
- Training task-database identity:
  `390a9d450f4e005051e4dccefa1287b42d6c51f8f5fed0f7ec06541b09faf03f`

## Runtime boundary

The V12 database declares task schema
`bosscrafts-current-task-resumable-v2`. The audited Mk10 runtime requires
`bosscrafts-current-task-resumable-v3`, which adds parser-runtime provenance.
The current runtime correctly rejected the historical database before model
loading with `task database schema mismatch`.

No missing provenance was invented and no current-lineage check was weakened.
The diagnostic used the preserved V12-era evaluator in read-only mode with
Python bytecode writes disabled. The checkpoint, copied databases, binding,
tokenizer, model fingerprint and V12 task-database identity were verified by
that historical evaluator before generation.

## Executed instruction diagnostic

- Device: NVIDIA GeForce RTX 3070
- Dtype: bfloat16
- Split: frozen V12 test
- Family: `instruction_to_code`
- Cases: 4
- Maximum new tokens: 128
- Exact matches: 0 / 4
- Protocol complete: 3 / 4
- Mean target-token accuracy: 0.1928896601000306
- Total generation time: 6.726765199913643 seconds
- Report SHA-256:
  `5fcce3348191920b7893d709e1d6a65a627ccd5c329268a8517b688038730fa3`
- Generation-records SHA-256:
  `e01d5ef7c9c0dc3c01e455de754872ea1643d65ab587493203139a63798b3b21`

The outputs were framed more coherently than the earlier recovered V8 smoke,
but the generated programs were incorrect. One case repeated invalid code
until the 128-token limit and failed to close its response protocol.

## Executed clean-core diagnostic

- Device: NVIDIA GeForce RTX 3070
- Dtype: bfloat16
- Split: frozen deterministic clean-v5 test
- Families: `code_to_semantic`, `semantic_to_code`, `semantic_continuation`
- Cases: 4 per family, 12 total
- Maximum new tokens: 128
- Exact matches: 0 / 12
- Protocol complete: 5 / 12
- Mean target-token accuracy: 0.16188752847076918
- Total generation time: 61.68176879978273 seconds
- Report SHA-256:
  `ff48fbc4808deefb1320f3d0a54eaf7b30c27844381f0d0d6417a67c4406131c`
- Generation-records SHA-256:
  `2ca3a8c124ec0d581d269f1f4421d641c01e4c5487258ef1a6b2163a2cf04aaa`

Per-family protocol completion was 0 / 4 for `code_to_semantic`, 4 / 4 for
`semantic_to_code`, and 1 / 4 for `semantic_continuation`.

## Decision

V12 is recovered and demonstrably loadable, but it is not a functioning or
promotable BossCrafts model. Completing its remaining 722 steps would continue
an unpromoted V11-derived lineage and cannot repair the missing current-lineage
parser provenance. V12 must remain preserved as diagnostic evidence, not used
as the parent for Mk10 training or release.

The model path returns to the audited Mk10 task-schema-v3 pipeline. New model
training remains gated on complete executable acceptance evidence and a frozen
current-lineage corpus rather than resuming V12.
