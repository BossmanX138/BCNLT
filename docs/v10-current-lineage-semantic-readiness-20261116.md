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

## Packed task databases

The validation and test partitions have completed packing with the pinned
`compact_v2` semantic wire and 2,048-token sequence bound. Their built-in full
database audits passed, and separately invoked full auditors reproduced the
same database identities, content hashes, counts, language distributions,
family distributions, split ownership, and token totals.

| Split | Documents | Examples | Sequence tokens | Packing rejects | Database identity SHA-256 | Content SHA-256 | Independent audit manifest SHA-256 | Receipt file SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| validation | 3,216 | 3,820,925 | 3,889,790,915 | 22,624 | `13a0970e8d5476d407e51eed8c298f11c2acb974dc883b4612c0798cb71b0f36` | `f23182857dc8d35a31d06de0414b2069ebdc181d605c17c0c982f3a54c801b41` | `86c02d5926c28ea156d7b215499527d574c5dbb63ea4e94f454f5d4e5229e1f2` | `2809b6788a5510feffc1f0b7794af15865d8df70dfc08c730fac7793f7da75bc` |
| test | 3,199 | 3,631,774 | 3,691,636,163 | 20,057 | `2bc3dd2b25e10a4fd87bc53a5f58a001791e4b01fe41130e9ee27ffca65d047f` | `c2fa59457a3000367d49678343ad0b424767a197ad742cf16669d61d8e49a520` | `dbf66ea3e883c42f954dfb78e5a89aeef3471cedeec74840b6bd50954452cd34` | `9612167a54664f28430e3d75eabb549413d4cd2baf74db1760a2473dd81c30de` |

Evidence roots:

- `D:\BCNLT-runs\tasks-current-8edc43e\validation\tasks.sqlite`
- `D:\BCNLT-runs\tasks-current-8edc43e\test\tasks.sqlite`
- `D:\BCNLT-runs\audits-current-8bdb702\validation-task-audit.json`
- `D:\BCNLT-runs\audits-current-8bdb702\test-task-audit.json`

The train task database remains resumable and in progress. It must complete its
own deferred cell index, built-in audit, and independently reproduced audit
before it can be used to build the mixed training curriculum.

## Reviewed instruction runtime repair

The prior `reviewed-instruction-v4` database retained parser-runtime identity
`7440ef62...`, so the repaired runtime correctly rejected it even though its
tokenizer and vocabulary identities still matched. The immutable reviewed
source at `D:\Wizard-Phi-3.5-v0.1\datasets\ready-v02` was rebuilt without new
semantic selection into:

- `D:\BCNLT-runs\curricula\reviewed-instruction-v5-runtime-repair\curriculum.sqlite`
- `D:\BCNLT-runs\audits-current-8bdb702\reviewed-instruction-v5-runtime-repair-task-audit.json`

The rebuild reproduces the prior content SHA-256
`bcd57661933cb186e7df19d45556bea9d0342e7d81a310e49cc72f0a5beec346`,
policy SHA-256
`e584b654f80c4ff038dac6deca3930bd977145bb9b2c6b751ee1cfac62a0ac46`,
57,943-example count, split counts, source counts, and rejection counts. Its new
database identity is
`3e9abdbcddd528f0bcb92ba5f383d45aeaa5ea61e9b31aeca9d35a054d1c185a`,
its parser-runtime identity is the current
`4cc9aac31f89be5c4e3f77ef61eabeea229dfa6db4c1dc80bd578a8f2d9a7917`,
and the independent audit passed with manifest SHA-256
`6135b95853ef2e23f0a5d674a473402bfcbf3736227a8db2ca556630b13fc6f8`.
The receipt file SHA-256 is
`a46f9d41fab6632e9749bb28a70c558f3728e7706fb7811963bbe8a069a6f7a3`.

## Remaining gate order

1. Finish packing the eligible train documents with the pinned `compact_v2`
   semantic wire and 2,048-token sequence bound, then reproduce its built-in
   full database audit with the independent auditor.
2. Audit every packed example's loss-mask boundary across the completed train,
   validation, and test task databases.
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
