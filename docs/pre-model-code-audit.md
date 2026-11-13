# Pre-model source audit and repair record

Date: 2026-11-13

Branch: `audit/v10-exhaustive-code-review`

This record covers the isolated BossCrafts model candidate and its contained
BCNLT v10 learning runtime. Model insertion and training remained held during
the review. The review does not claim that tests can prove the absence of every
possible future defect; it records that the reviewed source has no known
actionable defect after the repairs and gates below.

## Source and dependency gates

- Git and Git LFS integrity were established before repair work. The branch is
  stored remotely and reviewed through the repository rather than loose local
  files.
- The contained runtime is sealed by the normalized Python-source aggregate
  `c5623e97f3f1e1e884e1d54a316d174292fbcbf61da3ddfdbf3b432418e0450a`.
- The resulting parser runtime identity is
  `c1ca753bf4d1dbabf841fdcd120e5c3eec026b7211ca5c0f08d066074acee938`.
- Ruff passed every changed Python file.
- Mypy passed all 65 configured core source files and the changed orchestration
  scripts in strict mode.
- The latest complete regression passed: 541 tests and 195 subtests, with
  warnings treated as errors.
- The earlier semantic-integrity baseline measured 74.5463% branch coverage:
  17,861 covered and 4,539 missing lines; 3,252 covered and 2,670 missing
  branches. That measurement predates the later exhaustive hardening and is not
  presented as current coverage.
- `pip-audit` reported no known dependency vulnerabilities after upgrading the
  pinned PyTorch and setuptools chain.
- Bandit reported no medium- or high-severity findings. The remaining 82 low
  findings were reviewed as intentional subprocess use without a shell,
  non-secret protocol constants, destructor cleanup suppression, or
  deterministic non-cryptographic sampling.
- A comprehensive security scan inventoried 514 files and found one actionable
  resource-exhaustion path. That defect is included in the repairs below.

PyTorch emits one environment warning because optional NumPy is not installed.
No repository source uses NumPy conversion APIs, so this is not an exercised or
required runtime path. It remains an optional integration dependency rather
than a hidden functional failure.

## Repaired defects

1. Semantic-audit records could decompress a caller-selected SQLite zlib blob
   without an output bound. Decompression is now exact-stream and output
   bounded, with high-ratio, truncated, trailing, and concatenated stream
   regressions.
2. Wildcard Memory writers could overwrite evidence observed by another
   instance or process. Mutation now uses an in-process lock plus an operating
   system file lock, reloads under the lock, and flushes and fsyncs the ledger.
3. Extended Core activation and rollback could allocate duplicate activation
   sequence numbers. The complete read-check-write transaction is now locked
   across threads and processes.
4. Invalid or oversized local semantic-review responses were not durably
   recorded, allowing retries without complete failure lineage. The reviewer
   now stores content-addressed evidence and terminal failure entries, and it
   permits only literal loopback IP endpoints.
5. Docker cleanup launch failures and timeouts could escape the sandbox error
   contract. They are now converted to explicit cleanup failures.
6. Legacy non-resumable corpus admission and task packing could buffer an
   unbounded full run in memory. These paths are now bounded pilots and direct
   full work to the resumable implementations. The legacy `--full` switch fails
   closed with the correct replacement command.
7. Parser-provenance download redirects were not independently restricted.
   The origin, every redirect, and the final response must now remain on pinned
   HTTPS release hosts before hash-pinned content is accepted.
8. Locked PyTorch and setuptools versions had published advisories. The lock is
   upgraded to PyTorch 2.13.0 and setuptools 84.2.1, and the resulting complete
   dependency export passes `pip-audit`.
9. The parser subprocess could be replaced after the runtime inventory check
   without attesting the executable sidecar bytes in its response. Sidecar
   protocol V4 now reports its normalized source hash and complete parser-runtime
   identity, launch paths recheck the frozen source immediately before execution,
   and each parse rechecks its selected grammar before producing evidence.
10. Malformed semantic records could escape as ordinary validation exceptions
    even when the caller explicitly requested the auditable raw-code fallback.
    Semantic handoff validation now fails through the preprocessing error
    contract by default and uses raw-code fallback only when explicitly enabled,
    preserving the failure reason in the prepared input.
11. A training checkpoint recorded its training database but did not freeze the
    evaluation database and held-out suite chosen before training. Checkpoint V5
    now binds both identities, the bounded trainer validates the real manifest
    and database, and its V3 training plan includes the same immutable contract.
    Full-model training fails closed when that contract is absent.
12. Evaluation and inference evidence conflated the training database with the
    separate frozen evaluation database. Their new schemas preserve both
    identities, propagate the checkpoint-bound held-out manifest, and final
    acceptance rejects any post-training suite substitution while allowing a
    legitimate distinct training corpus.
13. A baseline evaluation could come from a different held-out suite than the
    candidate. Baseline loading now carries and validates the checkpoint-bound
    suite identity, and candidate evaluation requires an exact match.

## Boundaries

- Common Core and tokenizer IDs remain immutable. Learned meanings remain in
  Wildcard Memory and validated Extended Core versions.
- Generated corpus material remains outside Git; manifests, source, tests,
  locks, and small review artifacts belong in Git, while large model assets use
  Git LFS.
- Native parser binaries were verified by pinned inventory and hashes. This was
  a source and integration audit, not a machine-code decompilation audit.
- CodeRabbit CLI authentication is external to the repository and still
  requires an interactive user login. That does not affect the local gates or
  remote Git store, but an actual CodeRabbit review result must not be claimed
  until authentication and review complete.

## Current decision

The latest repair batch has no regression in the executed source and integration
gates, but the exhaustive review remains active and CodeRabbit review remains
pending authentication. Training and model insertion stay held. This record is
not evidence that a model has been trained, promoted, or improved.
