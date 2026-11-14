# BCNLT Native Extended Core Tokenizer Plan

1. Allocate deterministic native IDs after the exact BCNLT v1 vocabulary.
2. Preserve allocations across updates; tombstone revoked meanings and append new
   revisions without reuse or renumbering.
3. Emit and validate native identities in semantic streams while preserving exact
   canonical-record round trips.
4. Save and reload a nested, hash-bound tokenizer artifact.
5. Append deterministic embedding and LM-head rows while proving all prior rows
   remain exact and frozen.
6. Build a real artifact only from a frozen ledger containing scientifically
   validated entries.
7. Run held-out tokenizer and model compatibility gates before authorizing any
   bounded training run.
