# BossCrafts model architecture checkpoint

Date: 2026-11-12

This is an evidence checkpoint for the isolated V10 learning candidate. It is
not a model-promotion decision and does not authorize another training run.
The original BCNLT source, the R-drive tokenizer/runtime source, and the
verified BCNLM handoff archive were not modified by this work. Older V8 and
V12 training controllers that had auto-resumed were placed on an explicit
hold after their in-flight bounded chunks wrote durable checkpoints.

## Current contained runtime identity

- Runtime source files: 79
- Runtime source bytes: 705,094
- Runtime source aggregate SHA-256:
  `8f7f820ba20206a61ddc3509dd8c2a7564954bb3197de118fdc373a2fffc7c66`
- Grammar count: 14
- Grammar aggregate SHA-256:
  `19fc42827e8ab3b37282c0666caa1ddd8abcd3708298fb1665c10df440c3d090`
- Parser runtime identity SHA-256:
  `d0a0a61605faf57270d9df4828a01c281ace1af095e87c86fa2a5812b1cd950a`

## BCNLM handoff result

- The MSI archive was accessible and its supplied SHA-256
  `75f3956665d3f1382aa3bb9b6a2dadfa997b8aaa25e705ec5ab93425d4a0dfdd`
  was verified.
- The extracted handoff suite passed 83 tests.
- The repaired behavior passed against V10: 29 tests and 57 subtests.
- V10 already contained the repair behavior, so no older production source was
  copied into the later runtime.

## Implemented architecture

### Scientific execution sandbox

- One fresh container per request, no network, no host mounts, no devices, no
  published ports, read-only root, non-root UID/GID, all capabilities dropped,
  no-new-privileges, private cgroup namespace, no IPC, bounded CPU, memory,
  PIDs, output, filesystem, and time.
- The host validates the exact Docker inspect policy before execution and
  records image, policy, request, response, stdout, and stderr hashes.
- General image ID:
  `sha256:153a93c2ca76aacacdbdb7d84d4e1e63767c46ddd7e42f9957e9fe7f7c782f32`.
- Swift image ID:
  `sha256:7d9e1ed132050ad8f3e2aa0c6537f1609cf5b90532e273689c676d4fafec6262`.
- Real compile/execute coverage passes for Bash, C, C#, C++, Go, Java,
  JavaScript, Kotlin, PHP, Python, Ruby, Rust, Swift, and TypeScript.
- Sandbox-backed scientific evidence is schema 3. Unattested bounded JSON
  projection evidence cannot promote a learned meaning.

### Extended Core lifecycle

- Common Core identity is hashed and remains immutable.
- Only meanings with intact, independently reproduced, sandbox-backed evidence
  can become promotion candidates.
- Versions are content-addressed immutable directories with a full library
  snapshot, append-only vocabulary manifest, parent hash, evidence hashes, and
  version hash.
- Validation is a separate immutable receipt tied to exact version-file hashes.
- Activation uses an atomic pointer plus append-only activation events.
- Rollback activates the validated parent without deleting or rewriting either
  version.
- Previously allocated Extended Core token IDs remain unchanged when a later
  version appends another meaning.

### Uncertainty calibration

- Deterministic isotonic calibration is fit on an identified calibration split.
- Evaluation requires a disjoint held-out split and records exact data and
  manifest hashes.
- The contained calibration gate enforces Brier score, expected calibration
  error, total sample count, language count, and minimum samples per language.
- A separate BossCrafts strict receipt enforces global and per-language
  low-uncertainty selective coverage and selective error. Semantic-fidelity
  model acceptance now requires that passing receipt and its SHA-256.
- A failed, tampered, overlapping, or non-promoted artifact cannot wrap the
  runtime scorer.
- No real project calibration artifact has been promoted yet; the mechanism is
  implemented and regression-tested. The first real 140/140 disjoint label
  artifact is frozen, but it is deliberately not calibration-ready because all
  280 reviewed runtime records were kept and therefore neither split contains
  an observed semantic error. Both label classes are now required globally and
  in every language/split; a one-class artifact cannot report deceptively
  perfect calibration.

### Independent semantic review

- The current-runtime held-out admission contains 335 admitted documents and
  one rejected oversized/minified JavaScript document.
- The immutable review queue contains 280 records: 20 for each of all 14
  languages. Every item includes numbered source context and an exact target
  span. Its queue SHA-256 is
  `b7294cb3627618cb8d9c3c816bcdc9dbe3fcb3ff196f2d97e3aedb982bd273a5`.
- The dual-review runner accepts only loopback HTTP endpoints, requires two
  distinct reviewer/model identities, validates strict JSON decisions, stores
  content-addressed response evidence, resumes without duplicate requests,
  and routes disagreement or uncertainty to a manual queue.
- Invalid model responses receive at most one separately hashed repair request.
  Exhausted repairs become explicit human-review cases instead of looping.
  Transport failures are also hashed without inventing a response artifact.
- The completed run has 546 valid model decisions and 14 terminal exhausted
  repairs, covering all 560 reviewer slots. Consensus produced 237 agree-keeps,
  zero agree-rejects, 43 manual cases, and zero pending cases. All 43 manual
  cases were individually checked against source and adjudicated keep.
- Dual-review manifest SHA-256:
  `0f9ad6b7b2b6e20d61e1a216582a6783f5f533da18495983e5a9134119ab2edc`.
- Manual adjudication SHA-256:
  `8c2edf7cfaee4411a7beeb1955c7fa6075716a0a45a4cb1ab563e208247c0e13`.
- Dual-model agreement is only a proposed decision. It cannot set
  `semantic_review_passed` or `training_ready`; final adjudication remains a
  separate required step.

### Blinded reviewer qualification

- A separate 280-item challenge was derived from the verified natural queue as
  140 unchanged positive controls paired with 140 controlled defects. Its
  queue and answer key are separately hashed; expected decisions and defect
  types are never included in model requests.
- Challenge manifest SHA-256:
  `e63ced0de652f91a14032ed84ddfd58cf13030d287c3edabbcaa3165685d478b`.
  Queue SHA-256:
  `9fa5b0aa972125287db0f9a8f86864af7def90eead576db3758d3cad2658704e`.
  Sealed answer-key SHA-256:
  `1cfd2bdc6b921a8c73ead1b5d763a6e59a7bc180ff51ae583146d868d65d84d6`.
- The corrected balanced canary contains five matched positive/negative pairs,
  one for every defect type and five languages. Selection SHA-256:
  `9a0be6ee0df501c4a8bfe45772cee38190b34dc2ff531e991761c8228c616a4d`.
- The local Ornith IQ4_XS source GGUF was verified before registration. SHA-256:
  `0032d73f235213134c656eb155dd5e1fe7d05ae2f2d96a526c3273fd3cc09aec`.
- Reviewer protocols are versioned. Legacy evidence remains verifiable. V2 adds
  machine-extracted UTF-8 byte span evidence and explicit scope/uncertainty
  conventions. V3 also binds the pinned Common Coding Core event/IR contract
  and zero-width scope-boundary convention.
- The qualification gate is rubric-aware: a negative control counts only when
  the reviewer marks the rubric field corresponding to the planted defect
  false. Merely rejecting for an unrelated reason does not count.
- Under V3, Ornith detected all five controlled defect types, but kept only two
  of five unchanged controls: 7/10 overall, 1.0 negative sensitivity, and 0.4
  positive specificity. Runeforge detected none of the five defects and had one
  exhausted repair: 4/10 overall, 0.0 sensitivity, and 0.8 specificity.
- V3 review manifest SHA-256:
  `b6c7ec9a224d17966d82a5ca66966e7789febcdd7aa2af7b700fbbc67a54a708`.
  Rubric-aware qualification report SHA-256:
  `bb4b273bbe34c34b4930dd2a344d39af1e279df6fde61366646476b2ef52bbea`.
- Neither model is qualified as an autonomous semantic accept/reject authority.
  Ornith is useful as a defect-seeking reviewer, but its false-positive rate
  requires deterministic pre-auditing and final adjudication. The earlier
  all-keep dual-model natural review is therefore not strong independent
  validation by itself.

### Deterministic and full-source semantic gates

- A content-addressed deterministic pre-auditor now verifies exact UTF-8 span
  surfaces, the pinned Common Coding Core event/IR contract, module-rooted
  scopes, zero-width and source-anchored scope boundaries, and honest
  `UNKNOWN_SEMANTIC` uncertainty state. It never authorizes training.
- The natural 280-record queue produced zero findings. Natural audit manifest
  SHA-256:
  `57973e9c8ffb831df91348042a92add1389f1868ec4ae2eef16f1288fa1d74a8`.
  Its empty findings stream SHA-256 is the standard empty-content digest:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- On the 140 positive/140 negative blinded challenge, the finalized invariant
  gate flagged 111 negative controls and zero positive controls. In addition to
  span, surface, uncertainty, boundary, and event/IR-kind checks, non-extension
  IR names must match their claimed surfaces. Challenge audit manifest SHA-256:
  `996c9b41a4101df79ece09c88e27cdd750ebe205b1eec7dfd2542fcdc0be2fa4`.
  Findings SHA-256:
  `201a40a9ab90f41d4d5b8e6feed5025945d35fadc53d401fe896da4dce0d8861`.
- A separate raw-AST witness sidecar invokes Tree-sitter parsing but never the
  semantic extractor. It resolves declared nested field paths and synthetic
  identifier roles, independently derives lexical scope from enclosing syntax,
  and binds every report to its own source hash. Sidecar SHA-256:
  `eda5ab50e314f0355471f9f94a691703b9053edf7d6672b068fd09065644e77d`.
- All 280 natural controls pass the raw-AST relation/scope audit. Natural raw-AST
  manifest SHA-256:
  `60baf9f41c831b8252573754e92b22fdf5c5f834d35da6eec35e1748233f0298`.
  Witness stream SHA-256:
  `46dac12acc2ac55588161f6561f3ad6d22a02ef9e7421ae16d5b23d25caa5574`.
- The raw-AST gate independently flags 80 challenge defects: all 28 span swaps,
  all 28 scope swaps, and 24 of 28 event/IR swaps, with zero positive false
  flags. Challenge raw-AST manifest SHA-256:
  `f65cc034fbb31f4568ab0d44f537fc7f15747a5441cd9aebe4b124d404114564`.
  Witness stream SHA-256:
  `0bf959f3fdca1e44d67dbb936d64a04007e3873df4fcdc8e4f1bee9fa608ef39`.
- The deterministic and raw-AST gates together detect all 140 planted defects
  and flag none of the 140 unchanged controls. All five defect families reach
  100% controlled detection. The combined qualification was independently
  reverified and remains non-authorizing for training. Qualification report
  SHA-256:
  `0ad1c908a121ed8c9a43024a92c930939f5a7c30135a0490bc1923395a1cdc7b`.
- A separate full-source replay gate follows every review item back to the
  admitted source document, original relative path, dialect, and record index;
  it reparses with the pinned V10 sidecar and compares both the complete
  canonical stream and selected record. All 280 review items across 280 source
  documents replayed byte-for-byte with zero stream or record mismatches.
  Replay manifest SHA-256:
  `979aeecb7bbad9b9396a751db56d1cd402f0629ab8c651f88b92daadbc8d19f7`.
  Results SHA-256:
  `98322f6df34de2cecd9274d49302898a24c45da89cf34e1d8214a8d35565560d`.
- Replay proves current-runtime reproducibility. The raw-AST and invariant
  gates now provide an independent controlled oracle for the five frozen defect
  families; this does not imply completeness for every possible semantic bug.

### Leakage-resistant training source freeze

- The exact 241,963,008-byte source catalog used by the held-out admission is
  now copied into the isolated V10 candidate. Its SHA-256 remains
  `286d92f79f84b75049b845be0270e4308f32056421c8dea92d82d41b4e4003e2`;
  the preserved V4 through V7 copies were not modified.
- Every one of its 32,542 source documents was rechecked against its stored
  content SHA-256, approved SPDX license, attribution, and pinned source
  revision before freezing. The catalog contains 31,068 per-file licensed
  `bigcode/the-stack-dedup` documents and 1,474 CC-BY-4.0
  `nvidia/OpenCodeInstruct` documents.
- The earlier document-ID split placed 1,772 training documents in repositories
  that also appeared in validation or test. V10 now groups real Stack documents
  by repository, treats synthetic OpenCodeInstruct documents as independent
  document groups, and explicitly reserves every reviewed held-out document and
  its whole group.
- The frozen result contains 25,747 training documents, 3,219 validation
  documents, 3,207 test documents, and 369 reserved-held-out documents. All 336
  reviewed admission candidates are reserved, and real-repository crossover is
  zero.
- Freeze manifest SHA-256:
  `b1bdb44b55528e01b7d0c551ec9f6bf09092b0723226c1d508002f81fc490b57`.
  Frozen document inventory SHA-256:
  `16dd2e72c195791290835c18bf3937d13adb6550f1e7080ba1a9672b309f6788`.
  This is a source freeze only and remains `training_ready: false`.

## Verification

- Extended Core lifecycle: 2 tests passed using real sandbox reproductions.
- Local-only dual semantic review: 14 tests passed, including loopback
  subprocess/resume, bounded repair, exhausted-repair adjudication, evidence
  tamper detection, and transport-failure preservation.
- Semantic calibration dataset: 4 tests passed, including per-language,
  per-split label-class requirements.
- Complete V10 regression after deterministic auditing, raw-AST qualification,
  full-source replay, and source-freeze implementation:
  **304 tests and 133 subtests passed in 165.13 seconds**.
- No labeled sandbox containers remained after verification.

## Explicit training hold

No training worker or relaunch controller remained on either machine after the
in-flight bounded chunks drained. Training remains unauthorized until the
evidence gates below pass.

PATISBORG V8, which is incomplete and separate from V10:

- Optimizer steps at the latest durable checkpoint: 5,500 / 10,752 (51.2%).
- Examples seen: 44,000.
- Latest mean chunk loss: 1.248775100708008.
- Gradients finite: true.
- Checkpoint size: 3,240,810,635 bytes.
- Remaining optimizer steps: 5,252.

MSI V12 engineering line:

- Optimizer steps at the latest durable checkpoint: 17,200 / 17,922.
- Training complete: false.
- Checkpoint size: 3,240,810,635 bytes.

Neither older checkpoint is a current-V10-lineage accepted model.

## Remaining blockers before another training run

1. Run all 25,747 frozen training documents through current-runtime semantic
   admission, then run every admitted semantic record through the qualified
   deterministic plus raw-AST gates. The source inventory is now frozen and the
   280-record held-out queue passes, but the full training-record audit has not
   yet been executed.
2. Produce real class-diverse uncertainty calibration and held-out labels. The
   natural review queue contains no reject class, so it cannot calibrate
   real-world error probabilities even though controlled challenge detection is
   now perfect for its five frozen defect families. Current calibration manifest
   SHA-256 is
   `3d2374f65b85da65d4e5431ee6275573f61bdb6dc57d18c9abb65f492910f5b7`
   and it has 30 explicit class-diversity blockers.
3. Finish the frozen training configuration, resource bounds, resumable writes,
   and no-training dry run against only admitted examples.
4. After a new isolated candidate exists, generate the eight real, independently
   hashed acceptance artifacts for
   tokenizer fidelity, semantic fidelity, raw-code behavior, ordinary
   instruction following, sandbox execution correctness, safety, regression,
   and resource limits. The unified gate is implemented and cannot be bypassed
   by loss improvement alone.
5. Only after the pre-training gates pass, resume from a copied, verified checkpoint or start
   a new isolated candidate. Do not train directly into the preserved V8 file.

