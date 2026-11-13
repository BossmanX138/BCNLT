"""Build deterministic BCNLT recovery evidence from project-local copies only."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ROOT = PROJECT_ROOT / "artifacts" / "bcnlt" / "runtime-recovery-v1"
SOURCE_ROOT = RECOVERY_ROOT / "source"
CORPUS_ROOT = RECOVERY_ROOT / "admitted-corpus"
SPECS_PATH = RECOVERY_ROOT / "semantic-specs.json"
MANIFEST_PATH = RECOVERY_ROOT / "recovery-manifest.json"

TOKENIZER_SHA256 = "e1930449f3efe99bbb3f63730f75f71788d4ff21928bcae4d3ee2a520d30d407"
BINDING_SHA256 = "6bdcf5df4e5f16b79ee8de6eea9b86194c1e9909ef8e8545e28d5847ab031268"
PARSER_CONFIG_SHA256 = "d7df9e034b19a2a459299cecb1e14c31e4bd7b9626247f3ebd56ba258d3fecd6"
OBSERVED_SPEC_SHA256 = {
    "bash": "2456bb03da817645a32fe197ddd4a7fa1f9d39aa962462ad8ec2d36495105d72",
    "c": "4074bac723d50ecd2046329c537bef89977144935a948fa19828e67ddc6d0deb",
    "c_sharp": "8ea03777924d8ae3a4a6a90396f72fb4027c44528c5fa9e46fc2b790db0676d3",
    "cpp": "342de562401c299eefc900ac9ac69fd4890b59379aa680baf2c0cd7e3f307a",
    "go": "2701f4ebcf1be3d199ade39b26f443d9309d2d03f433e6d4083876cadfa758ee",
    "java": "e21c3fda821a487ef25868175d21a3d1f569c8de362c3ad77efe89b23ade5cf8",
    "javascript": "1fe022c4990ff4293e367bc4761b949ca9c5b5ba56493dec4c4fd06beddc9cb4",
    "kotlin": "1dbcf1aa8d4e9277b60e08790e97e19182cbdab2299125d6fd4d6ced00329547",
    "php": "0aa553d70c7fd1364eb8e884942908828a9b18a1b198f94b4376bf4ab5b5fc54",
    "python": "fe0bb47da50274411299c39d0034eb434c2d36fc1d58d42e4e9b847403b83303",
    "ruby": "9251389d48aa0f94a9ae19d6605b8dc306b2c0f1e834014ae09de6cd37c88148",
    "rust": "345d4da1650dfc4020aed3d86205f428882bb2350974dc6c01bf1a84478deef1",
    "swift": "2cb2c75ffa4bfe53c2936dba00461ccd25fc04f3dd9d63f97b66869fa67ba398",
    "typescript": "45b4d2a264fd9629ae5c6b00567a3d0d523fc8ef13d8646e32e3aa811a76f190",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: object) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def spec_payload(spec: object) -> dict[str, object]:
    return {
        "semantic_spec_version": spec.semantic_spec_version,
        "node_rules": [asdict(item) for item in spec.resolved_node_rules],
        "identifier_roles": [asdict(item) for item in spec.resolved_identifier_roles],
        "scope_rules": [asdict(item) for item in spec.resolved_scope_rules],
        "uncertainty_rules": [asdict(item) for item in spec.resolved_uncertainty_rules],
        "excluded_node_types": list(spec.resolved_excluded_node_types),
    }


def file_inventory(root: Path) -> dict[str, dict[str, object]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def main() -> None:
    sys.path.insert(0, str(SOURCE_ROOT))
    from bcnlt.common_core_registry import build_semantic_spec  # noqa: PLC0415
    from bcnlt.languages import CANONICAL_LANGUAGE_IDS  # noqa: PLC0415

    specs = {}
    comparisons = {}
    for language in CANONICAL_LANGUAGE_IDS:
        spec = build_semantic_spec(language)
        payload = spec_payload(spec)
        digest = canonical_hash(payload)
        if digest != spec.canonical_hash:
            raise RuntimeError(f"semantic spec canonicalization mismatch for {language}")
        specs[language] = {"canonical_sha256": digest, "payload": payload}
        comparisons[language] = {
            "recovered_source_sha256": digest,
            "admitted_output_sha256": OBSERVED_SPEC_SHA256[language],
            "matches": digest == OBSERVED_SPEC_SHA256[language],
        }

    specs_document = {
        "schema_version": "bosscrafts-bcnlt-semantic-spec-snapshot-v1",
        "source_head": "4857b8ec0d765c4ecf50da352aa610b1571c0ffb",
        "specs": specs,
    }
    SPECS_PATH.write_text(
        json.dumps(specs_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    packed = json.loads((CORPUS_ROOT / "packed--dataset_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((CORPUS_ROOT / "per-language-validation.json").read_text(encoding="utf-8"))
    recovery = json.loads(
        (CORPUS_ROOT / "corpus-validation-recovery-gate-v2.json").read_text(encoding="utf-8")
    )
    source_sample = json.loads((CORPUS_ROOT / "source-sample.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "bosscrafts-bcnlt-runtime-recovery-v1",
        "source": {
            "origin": r"\\PATISBORG\C$\Users\bossm\Documents\Codex\BCNLT\.worktrees\bcnlt-multilingual-corpus",
            "branch": "feature/bcnlt-multilingual-corpus",
            "head": "4857b8ec0d765c4ecf50da352aa610b1571c0ffb",
            "working_tree_reported_dirty": True,
            "identity_policy": "exact_current_file_hashes_override_commit_assumptions",
            "files": file_inventory(SOURCE_ROOT),
        },
        "parser_contract": {
            "semantic_schema_version": "bcnlt-semantic-record-v3",
            "parser_config_sha256": PARSER_CONFIG_SHA256,
            "grammar_package": "tree-sitter-language-pack==1.22.1",
            "tree_sitter_package": "tree-sitter==0.26.0",
            "max_source_bytes": 204800,
            "parse_timeout_ms": 500,
            "semantic_specs_file": SPECS_PATH.name,
            "semantic_specs_file_sha256": sha256(SPECS_PATH),
            "semantic_spec_comparison": comparisons,
            "all_specs_match_admitted_output": all(item["matches"] for item in comparisons.values()),
        },
        "runtime": {
            "source_environment_python": "3.13.13",
            "project_python": "3.12.10",
            "source_native_package_sha256": "18fda57fbe2f52f85cf09fd1fc5af6509bfe0a8c6d3d425cafe3d7b63f8dca6e",
            "source_tree_sitter_binding_sha256": "d8efa5533a1c79cc404df3a0ff03d76626ad50d54b74d97e4cf6e30ab85bebe4",
            "native_package_copied": False,
            "native_abi_compatible": False,
            "grammar_cache_complete": False,
            "grammar_dlls_found": 0,
            "grammar_dlls_required": 14,
            "automatic_download_permitted": False,
            "live_parser_executed": False,
            "metadata_files": file_inventory(RECOVERY_ROOT / "runtime-metadata"),
        },
        "admitted_corpus": {
            "origin": r"\\PATISBORG\D$\BCNLT-runs\runeforge-mk2-bcnlt-multilingual-v1\multilingual-smoke-21b9de0343dd4bb2-gate-v2",
            "bulk_accepted_jsonl_copied": False,
            "bulk_accepted_jsonl_bytes": 5318202517,
            "accepted_record_count": recovery["final_accepted_record_count"],
            "packed_training_records": audit["training_records"],
            "packed_training_tokens": audit["training_tokens"],
            "languages_passed": len(audit["language_reports"]),
            "source_sample_passed": source_sample["passed"],
            "packed_manifest_sha256": sha256(CORPUS_ROOT / "packed--dataset_manifest.json"),
            "packed_tokenizer_vocab_size": packed["tokenizer_vocab_size"],
            "packed_tokenizer_manifest_sha256": packed["tokenizer_manifest_sha256"],
            "evidence_files": file_inventory(CORPUS_ROOT),
        },
        "frozen_model_identity": {
            "tokenizer_vocab_size": 65536,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "binding_sha256": BINDING_SHA256,
        },
        "readiness": {
            "source_contract_recovered": True,
            "output_consumer_integrated": True,
            "live_parser_ready": False,
            "admitted_corpus_tokenizer_compatible": packed["tokenizer_vocab_size"] == 65536,
            "training_ready": False,
            "blockers": [
                "14 pinned grammar DLLs are absent",
                "available native parser package is CPython 3.13 while project Python is 3.12",
                "all 14 recovered semantic spec hashes differ from admitted output provenance",
                "admitted packed corpus uses tokenizer vocab 32073, not frozen vocab 65536",
            ],
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"manifest": str(MANIFEST_PATH), "sha256": sha256(MANIFEST_PATH)}))


if __name__ == "__main__":
    main()
