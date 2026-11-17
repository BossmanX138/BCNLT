# Mk10 runtime / preserved V8 checkpoint read-only smoke

Date: 2026-11-13

## Purpose

Test whether the current audited Mk10 runtime can load and execute the only
locally preserved BossCrafts checkpoint without modifying that checkpoint or
claiming it as current-lineage Mk10 weights.

## Preserved input

- Source checkpoint:
  `C:\Users\bossm\OneDrive\Documents\ChatGPT\BCNLT\.codex_staging_bosscrafts_model_v8_candidate\artifacts\checkpoints\bosscrafts-engineering-v8\checkpoint.pt`
- Size: 3,240,810,635 bytes
- SHA-256: `a8b52d9be124d4ae0f68c1a4745d4edc3ce894e34a05228f15512b1a29f83a0a`
- Durable optimizer step: 5,500 / 10,752
- Training complete: false
- Checkpoint task-database identity:
  `f66255dbee97826a604819f8db2f71ec971c521a8d232887e02703fc9e626c52`
- The checkpoint was read in place. No source artifact was written or copied.

## Compatibility verified

- Current binding SHA-256:
  `6bdcf5df4e5f16b79ee8de6eea9b86194c1e9909ef8e8545e28d5847ab031268`
- Current tokenizer SHA-256:
  `e1930449f3efe99bbb3f63730f75f71788d4ff21928bcae4d3ee2a520d30d407`
- Current model fingerprint:
  `383bc3acb9bea01ed9a40fb6293a5ac81ee7ffb27ffbc4fdcfeafd96358b3bca`
- Parameter count: 540,094,726
- The checkpoint loader verified the complete checkpoint hash, finite model and
  optimizer state, exact binding, model fingerprint and task-database lineage
  before restoring the model.

## Executed smoke

- GPU: NVIDIA GeForce RTX 3070 8 GB
- CUDA interpreter: PyTorch 2.11.0+cu130
- Dtype: bfloat16
- Task: `instruction_to_code`
- Prompt: `Write a Python function named add that returns the sum of two numbers.`
- Maximum new tokens: 32
- Actual new tokens: 8
- Elapsed generation time: 0.73064 seconds
- Peak allocated CUDA memory: 1,097,457,152 bytes
- Native response protocol complete: true
- Generated protocol text:
  `<|bcnlt:stream:begin|><|bcnlt:mode:raw_code|>\t\t\t_,\n<|bcnlt:stream:end|><|bcnlt:task:end|>`
- Decoded payload: `\t\t\t_,\n`

## Decision

The Mk10 runtime, native tokenizer binding, checkpoint loader, GPU execution
and response framing work together on real preserved weights. The generated
payload is not a valid answer, so this checkpoint is not a functioning
instruction model and must not be promoted. This is consistent with its
incomplete V8 training state and does not diagnose a tokenizer or runtime-load
failure.

The next meaningful model test requires either a completed compatible
checkpoint with its frozen evaluation lineage, or a new isolated candidate
trained only after the remaining pre-training gates are complete.
