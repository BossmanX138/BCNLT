# V10 current-lineage semantic readiness, 2026-11-16

## Scope

This receipt records the fresh admission-bound semantic audits and eligibility
partitions produced after commit `fa4c826a4137f2f101b5bbc5e3970a89f5ba5f7c`
normalized CRLF inside raw-AST-derived scope names. It does not authorize model
training by itself. Task packing, loss-mask auditing, calibration, and the final
promotion gate remain separate evidence requirements.

The older discovery database at
`D:\BCNLT-runs\semantic-audit-current-4cc9aa\train\audit.sqlite` is retained for
diagnosis only. Although it contains a completed traversal and 554 CRLF-only
scope findings, its final host-code immutability check failed because the repair
was made during that invocation. It must not be used for promotion or resumed
into the current lineage.

## Current semantic audits

All three current audits exited successfully after verifying that their pinned
host-code inventories were unchanged throughout each invocation.

| Split | Source documents | Admitted | Source rejects | Semantic records | Findings | Failed records | Audit database SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 25,747 | 25,711 | 36 | 13,486,782 | 0 | 0 | `bb2351120075077f2330179db7221b0c253335128136f1f427993bb8afb0e23f` |
| validation | 3,219 | 3,216 | 3 | 1,762,805 | 0 | 0 | `e722f3ee2924f9a3f636354420b548dff66b6270babb25886f5295735f6a5942` |
| test | 3,207 | 3,199 | 8 | 1,673,365 | 0 | 0 | `a1457ae7d3338703fc99546b42b5cf4dfc054910c020a822f24fc07488f2570d` |

Evidence roots:

- `D:\BCNLT-runs\semantic-audit-current-fa4c826\train`
- `D:\BCNLT-runs\semantic-audit-current-fa4c826\validation`
- `D:\BCNLT-runs\semantic-audit-current-fa4c826\test`

The training audit host-code identity is
`9caca4faadbb75cefbec9f3f562f724db7e4f080cffa0489589eff64b759a1fb`.
The evaluation audit host-code identity is
`5c48cc40f51344497c474de3004d146837e7f8080a13b37aff188334c6cfdd3d`.
They differ because the evaluation controller is itself part of the evaluation
host-code inventory; they bind to the same repaired parser runtime and raw AST
sidecar.

## Eligibility partitions

Eligibility was materialized only from the zero-finding audits above. Each
partition exactly accounts for its admission database as eligible plus excluded
documents.

| Split | Eligible | Excluded | Manifest identity SHA-256 | Manifest file SHA-256 |
| --- | ---: | ---: | --- | --- |
| train | 25,711 | 36 | `55cc88d5a6bc9653656e68df27a8a559931640d5829440e7f26ab15d7b094af0` | `faf9e2d801621c2e1e9b4c350d79eda1b48946e8488432e7c1f99361641c6a92` |
| validation | 3,216 | 3 | `9a3d90274654b414621e0521d6989ce027bc87e5a2762a747281cff0703a3807` | `13ad7425a7d23df45648b4d7cafbd5c368fe286978ea9ddba8f76f5119726776` |
| test | 3,199 | 8 | `6972efe776fd430b088bf986dd80a2776b3cbeb6aeac68f43d0c07ea464be62e` | `70bbcc33f462657727fa8047dd8b1386f577bb96a9200f827ba9f0d392145f4b` |

Evidence roots:

- `D:\BCNLT-runs\eligibility-current-fa4c826\train`
- `D:\BCNLT-runs\eligibility-current-fa4c826\validation`
- `D:\BCNLT-runs\eligibility-current-fa4c826\test`

## Remaining gate order

1. Pack the eligible train, validation, and test documents with the pinned
   `compact_v2` semantic wire and 2,048-token sequence bound.
2. Audit every packed example and its loss-mask boundary across all three task
   databases.
3. Build a train-only, split-safe semantic and reviewed-instruction curriculum
   from the current training-task identity. Copy only reviewed instruction
   `train` rows; retain semantic validation/test databases and reviewed
   instruction validation/test rows as held-out artifacts with separate
   identities. Do not recombine held-out rows into the training database or
   reuse an older packed database.
4. Produce genuine adversarial calibration qualification plus a disjoint
   natural zero-error sample; no synthetic or fabricated reject labels may be
   substituted.
5. Run the final readiness and promotion checks before any model training.

Every artifact in this receipt intentionally keeps `training_ready=false`.
