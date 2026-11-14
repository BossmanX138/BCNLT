from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINER_SCHEMA_VERSION = "bcnlt-native-foundation-byte-bpe-v1"
STANDARD_SPECIAL_TOKENS = ("<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>")


def train_byte_bpe_tokenizer(
    *,
    corpus_files: tuple[str | Path, ...],
    structural_tokens_path: str | Path,
    output_dir: str | Path,
    vocabulary_size: int,
    min_frequency: int,
    threads: int,
    label: str,
) -> dict[str, Any]:
    if vocabulary_size < 512:
        raise ValueError("vocabulary_size must be at least 512")
    if min_frequency < 1 or threads < 1:
        raise ValueError("min_frequency and threads must be positive")
    files = tuple(Path(path) for path in corpus_files)
    if not files or any(not path.is_file() or path.stat().st_size == 0 for path in files):
        raise ValueError("every corpus file must be a non-empty file")

    structural_path = Path(structural_tokens_path)
    structural_payload = json.loads(structural_path.read_text(encoding="utf-8"))
    structural_tokens = structural_payload.get("tokens") if isinstance(structural_payload, dict) else None
    if not isinstance(structural_tokens, list) or not all(
        isinstance(token, str) and token for token in structural_tokens
    ):
        raise ValueError("structural token artifact must contain non-empty string tokens")
    if len(set(structural_tokens)) != len(structural_tokens):
        raise ValueError("structural tokens must be unique")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer_path = destination / "tokenizer.json"
    manifest_path = destination / "manifest.json"
    progress_path = destination / "progress.json"
    if tokenizer_path.exists() or manifest_path.exists():
        raise FileExistsError(f"tokenizer output already exists: {destination}")

    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

    corpus_bytes = sum(path.stat().st_size for path in files)
    started_at = _now()
    start = time.monotonic()
    _atomic_write_json(
        progress_path,
        {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "status": "running",
            "stage": "training",
            "label": label,
            "host": socket.gethostname(),
            "vocabulary_size": vocabulary_size,
            "threads": threads,
            "corpus_bytes": corpus_bytes,
            "started_at": started_at,
            "updated_at": _now(),
        },
    )
    try:
        unk_token = STANDARD_SPECIAL_TOKENS[0]
        tokenizer = Tokenizer(models.BPE(unk_token=unk_token, byte_fallback=True))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        special_tokens = [
            AddedToken(token, special=True, normalized=False)
            for token in (*STANDARD_SPECIAL_TOKENS, *structural_tokens)
        ]
        trainer = trainers.BpeTrainer(
            vocab_size=vocabulary_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train([str(path) for path in files], trainer)
        actual_vocab = tokenizer.get_vocab_size()
        if actual_vocab != vocabulary_size:
            raise RuntimeError(
                f"trainer produced {actual_vocab} tokens, expected exactly {vocabulary_size}"
            )
        temporary_tokenizer = tokenizer_path.with_name(f".{tokenizer_path.name}.tmp")
        tokenizer.save(str(temporary_tokenizer), pretty=False)
        temporary_tokenizer.replace(tokenizer_path)

        roundtrip_samples = (
            "plain English and code: value += 1;",
            "NUL:\x00 accents:café scripts:東京 emoji:🚀\nline two",
        )
        for sample in roundtrip_samples:
            encoded = tokenizer.encode(sample)
            decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
            if decoded != sample:
                raise RuntimeError(f"round-trip validation failed for {sample!r}")
        for token in structural_tokens:
            if len(tokenizer.encode(token).ids) != 1:
                raise RuntimeError(f"structural token is not atomic: {token}")

        elapsed = max(time.monotonic() - start, 1e-9)
        manifest: dict[str, Any] = {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "status": "complete",
            "label": label,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "vocabulary_size": actual_vocab,
            "min_frequency": min_frequency,
            "threads": threads,
            "byte_fallback": True,
            "pre_tokenizer": "ByteLevel(add_prefix_space=False,use_regex=True)",
            "decoder": "ByteLevel",
            "structural_token_count": len(structural_tokens),
            "standard_special_tokens": list(STANDARD_SPECIAL_TOKENS),
            "corpus_files": [str(path) for path in files],
            "corpus_bytes": corpus_bytes,
            "corpus_sha256": {path.name: _sha256(path) for path in files},
            "structural_tokens_sha256": _sha256(structural_path),
            "tokenizer_sha256": _sha256(tokenizer_path),
            "started_at": started_at,
            "completed_at": _now(),
            "elapsed_seconds": round(elapsed, 3),
            "input_megabytes_per_second": round(corpus_bytes / (1024 * 1024) / elapsed, 3),
        }
        _atomic_write_json(manifest_path, manifest)
        _atomic_write_json(progress_path, {**manifest, "stage": "complete", "updated_at": _now()})
        return manifest
    except Exception as error:
        _atomic_write_json(
            progress_path,
            {
                "schema_version": TRAINER_SCHEMA_VERSION,
                "status": "failed",
                "stage": "failed",
                "label": label,
                "host": socket.gethostname(),
                "vocabulary_size": vocabulary_size,
                "threads": threads,
                "corpus_bytes": corpus_bytes,
                "started_at": started_at,
                "updated_at": _now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))
