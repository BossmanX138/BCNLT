# BCNLT Native Extended Core Vocabulary

## Goal

Give scientifically validated Extended Core meanings dedicated model-facing token
IDs without changing the preserved base tokenizer, the frozen BCNLT v1 structural
inventory, or the immutable Common Core.

## Vocabulary layers

The authoritative order is:

1. preserved base-model tokens;
2. frozen BCNLT v1 structural tokens;
3. append-only native Extended Core tokens;
4. optional frequency-derived adaptive macros built against that exact parent.

Extended Core allocation is evidence-driven, not corpus-frequency-driven. A token
is allocated only for a currently validated `ExtendedCoreEntry`. Initial allocation
sorts stable Extended Core IDs. Later manifests preserve every existing name, ID,
definition, and model row and append new identities at the end.

## Revocation and revision

Revocation never deletes or reuses an ID. The token becomes an inactive tombstone:
new encodes cannot emit it, but historical streams remain decodable. A corrected
hypothesis revision has a new Extended Core ID and receives a newly appended token
and embedding/output row.

## Encoding contract

For a validated Extended Core semantic record, the native token appears immediately
after the record event token. The canonical Extended Core metadata remains in the
record as an auditable decoding contract. Decode requires the native token and
metadata identity to agree; missing, misplaced, inactive-on-encode, or mismatched
tokens fail closed.

Text and raw-code modes remain byte-for-byte on the v1 path and cannot contain
Extended Core tokens.

## Model-row contract

Native token rows append to the existing sparse BCNLT embedding and LM-head
extensions. All preserved base rows and all prior BCNLT rows remain exact and
frozen. New rows receive deterministic initialization from the manifest definition
and recorded seed. Tombstone rows remain in the tables.

## Artifact and promotion boundary

The native artifact nests an unchanged v1 tokenizer artifact and a canonical,
hash-validated Extended Core vocabulary manifest. The manifest binds the exact v1
manifest, parent manifest, active evidence snapshot, stable identities, allocation
order, and vocabulary size.

This phase establishes allocation, round-trip, artifact reload, and row-growth
compatibility. It does not fabricate validated meanings, train model weights,
merge a checkpoint, or promote a tokenizer. Training becomes eligible only when a
real frozen evidence ledger contains validated entries and the resulting artifact
passes held-out tokenizer, semantic, raw-code, ordinary-text, and model-row gates.
