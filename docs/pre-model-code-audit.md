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
- Mypy passed all 71 configured core source files and the changed orchestration
  scripts in strict mode.
- The latest complete combined regression passed 582 tests and 195 subtests
  with warnings treated as errors.
- The complementary Docker containment slice passed all 13 Docker-only tests
  and 13 subtests; 560 other tests were explicitly deselected.
- The exact Google IFEval revision
  `966cd89545d6b6acfd7638bc708b98261ca58e84` froze 541 executable-constraint
  cases from source SHA-256
  `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2`.
- The official Google IFEval evaluator is installed from exact revision
  `e6890f85757dd84e27ca6df2dd30651dafad28e0` with every source byte pinned.
  Its isolated runtime also freezes the required NLTK `punkt` and `punkt_tab`
  archives at revision `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`.
  The installed 129-file, 66,482,565-byte runtime verified as manifest
  `44f4d751b203a2c8330542e00d8abb418e270f00697c2a4ba5badfd18abd6712`.
- The exact NVIDIA Aegis 2.0 revision
  `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c` froze 1,928 reconstructable
  held-out prompt-label cases from test-source SHA-256
  `b0a6d602260524866053cb34105194f074f2c2906e3691b68d43b9e6e9318f35`;
  36 externally redacted prompts were deliberately excluded.
- The earlier semantic-integrity baseline measured 74.5463% branch coverage:
  17,861 covered and 4,539 missing lines; 3,252 covered and 2,670 missing
  branches. That measurement predates the later exhaustive hardening and is not
  presented as current coverage.
- A fresh locked production dependency export passed `pip-audit` 2.9.0 with no
  known vulnerabilities after upgrading the pinned PyTorch and setuptools
  chain.
- Bandit reported no unsuppressed medium- or high-severity findings. Its one
  low-confidence medium SQL-construction warning was traced to a bounded count
  of literal SQLite `?` placeholders; every validated SHA-256 ID remains a
  bound parameter, and the exception is documented in the source. Remaining
  low findings were reviewed as intentional subprocess use without a shell,
  non-secret protocol constants, destructor cleanup suppression, or
  deterministic non-cryptographic sampling.
- A comprehensive security scan inventoried 514 files and found one actionable
  resource-exhaustion path. That defect is included in the repairs below.
- An expanded Ruff diagnostic pass over security, Bugbear, pylint error/warning,
  simplification, performance and Ruff-specific rules found no Bugbear or
  pylint error/warning defects. Security reports were re-traced to fixed
  `shell=False` argument vectors, bound SQL values with count-only generated
  placeholders, and a URL opener guarded by literal-loopback-only validation.

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
14. Category case inventories could mix test-database identities or count the
    same packed task more than once under different provenance. Inventory V2
    now freezes one test-database identity and exact packed-task references;
    held-out publication proves every referenced task exists in the audited
    test database with the declared split, language and family.
15. The Project CodeNet pilot writer could overwrite prior evidence, expose a
    partial corpus/report pair after interruption, and publish an empty pilot
    before failing. Corpus and report now publish as one staged, synced,
    no-clobber generation and reject empty or count-mismatched inputs before
    creating either artifact.
16. Category freezing preserved case IDs but not the case payloads required to
    execute those evaluations later. The freeze command now publishes all eight
    validated JSONL catalogs, Inventory V2 and a self-hashed bundle manifest as
    one immutable generation. Reloading requires an exact file inventory,
    verifies every byte hash and count, and rebuilds the content-addressed
    inventory from the preserved cases before accepting the bundle.
17. Ordinary-instruction cases assumed one exact response string, safety cases
    carried only a free-form policy/action pair, and regression cases carried
    an arbitrary command string. Those shapes could not reproduce IFEval,
    preserve human Aegis labels, or support a safe fixed runner. Category Case
    V2 now stores IFEval instruction IDs and arguments, exact Aegis held-out
    prompt labels and taxonomy, a fixed regression-suite identifier, and
    revision, file-hash, license and attribution provenance. Offline no-clobber
    freezers validated the pinned real sources. Their real-data smoke exposed
    and repaired two additional assumptions: official IFEval rows can repeat an
    instruction type with different arguments, and the released Aegis test
    taxonomy and safe-label category combinations differ from the prose list.
18. Ordinary-instruction acceptance still consumed caller-asserted booleans,
    so a fabricated record could claim task success without executing the
    frozen IFEval constraints. The pinned official evaluator now runs in an
    isolated interpreter with a restricted environment, bounded payloads,
    deterministic language detection, exact runtime verification before and
    after execution, and strict case/response inventory checks. Acceptance now
    measures official strict and loose instruction following, baseline delta,
    and empty-response rate. A real all-541-case diagnostic run completed in
    2.298 seconds and left the pinned runtime unchanged.

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
- The canonical review surface is draft PR
  `https://github.com/BossmanX138/BCNLT/pull/2`; its head follows the audit
  branch automatically.

## Current decision

The latest repair batch has no regression in the executed source and integration
gates, but the exhaustive review remains active and CodeRabbit review remains
pending authentication. The ordinary-instruction runner is executable and
audited, but its batch output is not yet published as an immutable receipt and
the acceptance compiler does not yet require that receipt. The safety and other
remaining category runners are also incomplete. Training and meaningful model
insertion stay held; a read-only diagnostic Mk10 smoke would be permissible but
would not be promotion evidence. This record is not evidence that a model has
been trained, promoted, or improved.
