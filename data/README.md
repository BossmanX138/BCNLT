# BCNLT semantic-pair database

`bcnlt_semantic_pairs.sqlite3` is the local, versioned SQLite store for BCNLT semantic training candidates. It preserves source documents, contextual units, paired raw and BCNLT contexts, expected definition-line answers, uncertainty scores, provenance, review state, evaluation holdouts, and observed model answers.

Completed comparison runs can be appended without rebuilding the pairs database:

```powershell
python scripts/import_bcnlt_observations.py outputs/head-to-head-wizard-v02-vs-runeforge-mk2/results.json
```

The generated database, JSONL export, and manifest remain local artifacts. Rebuild them deterministically with:

```powershell
python scripts/build_semantic_pair_database.py `
  --holdout-list data/head-to-head-holdout.txt `
  --observations outputs/head-to-head-wizard-v02-vs-runeforge-mk2/results.json
```

The JSONL export contains every current pair and carries its admission fields. New records start with `review_status = unreviewed` and `training_eligible = false`. A future training job must select only explicitly reviewed, eligible records and must exclude `evaluation_holdout` rows.

Main tables:

- `documents`: source text, path, hash, split, and holdout reason.
- `semantic_units`: BCNLT events, spans, symbol links, IR payloads, and uncertainty.
- `semantic_pairs`: the question, raw context, BCNLT context, expected answer, provenance, and admission state.
- `model_observations`: per-model answers and correctness for both context forms.
- `ingestion_runs`: repeatable source-manifest snapshots and generator versions.
