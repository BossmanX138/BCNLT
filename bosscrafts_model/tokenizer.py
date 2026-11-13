"""Strict frozen BCNLT byte-BPE binding for the 65K model lineage.

This adapter performs raw text/byte-BPE tokenization. It can round-trip a
serialized semantic record, but it does not parse or produce BCNLT semantics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TOKENIZERS_RUNTIME_ROOT = _PROJECT_ROOT / "vendor" / "tokenizers_runtime_0_23"
_OWNED_TOKENIZERS_MODULE = "_bosscrafts_owned_tokenizers_0_23_2"
EXPECTED_RUNTIME_BINARY_SHA256 = (
    "05848a2a85e1466d19242499ed6b8c016a886db79aca020792aa5598ded77ef9"
)
EXPECTED_RUNTIME_PROVENANCE_SHA256 = (
    "5ec01ac2e8884a06a3c77d274a968ad400dcf8db3ef26644ed06c68e2412f0ca"
)
EXPECTED_RUNTIME_TREE_SHA256 = (
    "8debceae31cb21378fb540a16987905c81095cfc8c8fd1f10a67175820480d8e"
)
EXPECTED_RUNTIME_FILE_COUNT = 34
EXPECTED_RUNTIME_BYTES = 8_241_791

if TYPE_CHECKING:
    from tokenizers import Tokenizer as TokenizerType

    from .config import ModelConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_tree_identity(root: Path) -> tuple[str, int, int]:
    """Hash all executable/distributed runtime files, ignoring bytecode caches."""

    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name != "PROVENANCE.json"
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\t{_sha256(path)}\n".encode())
        total_bytes += path.stat().st_size
    return digest.hexdigest(), len(paths), total_bytes


def _verify_owned_tokenizers_runtime(root: Path) -> None:
    provenance = root / "PROVENANCE.json"
    if (
        not provenance.is_file()
        or _sha256(provenance) != EXPECTED_RUNTIME_PROVENANCE_SHA256
    ):
        raise ValueError("owned tokenizers runtime provenance identity mismatch")
    identity, file_count, total_bytes = _runtime_tree_identity(root)
    if (
        identity != EXPECTED_RUNTIME_TREE_SHA256
        or file_count != EXPECTED_RUNTIME_FILE_COUNT
        or total_bytes != EXPECTED_RUNTIME_BYTES
    ):
        raise ValueError("owned tokenizers runtime tree identity mismatch")


def _load_owned_tokenizers() -> ModuleType:
    """Load the pinned native package privately, independent of import order."""

    runtime_root = _TOKENIZERS_RUNTIME_ROOT.resolve()
    _verify_owned_tokenizers_runtime(runtime_root)
    package_root = (runtime_root / "tokenizers").resolve()
    runtime_binary = package_root / "tokenizers.pyd"
    if (
        not runtime_binary.is_file()
        or _sha256(runtime_binary) != EXPECTED_RUNTIME_BINARY_SHA256
    ):
        raise ValueError("owned tokenizers runtime binary SHA-256 mismatch")
    existing = sys.modules.get(_OWNED_TOKENIZERS_MODULE)
    if isinstance(existing, ModuleType):
        return existing
    spec = importlib.util.spec_from_file_location(
        _OWNED_TOKENIZERS_MODULE,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("owned tokenizers runtime could not be loaded")
    module = importlib.util.module_from_spec(spec)
    # The upstream wrapper imports some extension namespaces through absolute
    # names (for example ``tokenizers.decoders``).  Snapshot the entire public
    # namespace, not only its root, so the private load cannot leak owned
    # submodules into a caller that later imports another tokenizers runtime.
    previous_public_modules = {
        name: value
        for name, value in sys.modules.items()
        if name == "tokenizers" or name.startswith("tokenizers.")
    }
    for name in previous_public_modules:
        sys.modules.pop(name, None)
    sys.modules[_OWNED_TOKENIZERS_MODULE] = module
    # The upstream package contains absolute imports such as
    # ``from tokenizers import Encoding`` inside otherwise relative modules.
    # Supply the verified owned package under that public name only while its
    # wrapper executes, then restore the caller's import state.  Without this
    # alias, a clean environment fails to import the binding; with a permanent
    # alias, the binding would unexpectedly replace a caller's package.
    sys.modules["tokenizers"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_OWNED_TOKENIZERS_MODULE, None)
        raise
    finally:
        for name in tuple(sys.modules):
            if (
                name == "tokenizers" or name.startswith("tokenizers.")
            ) and name not in previous_public_modules:
                sys.modules.pop(name, None)
        # `sys.modules` can contain deliberate None sentinels at runtime,
        # although its static type exposes only ModuleType values. Preserve the
        # caller's exact namespace rather than silently converting it.
        for name, value in previous_public_modules.items():
            sys.modules[name] = value
    return module


# The private name prevents an earlier site-packages import from selecting the
# native implementation used by this binding.
tokenizers = _load_owned_tokenizers()


EXPECTED_TOKENIZER_SHA256 = (
    "e1930449f3efe99bbb3f63730f75f71788d4ff21928bcae4d3ee2a520d30d407"
)
EXPECTED_BINDING_SHA256 = (
    "6bdcf5df4e5f16b79ee8de6eea9b86194c1e9909ef8e8545e28d5847ab031268"
)
EXPECTED_VOCABULARY_SIZE = 65_536
EXPECTED_RUNTIME_MAJOR_MINOR = "0.23"
EXPECTED_RUNTIME_VERSION = "0.23.2"
DEFAULT_BUNDLE = (
    _PROJECT_ROOT / "artifacts" / "tokenizers" / "bcnlt-native-foundation-65536-v1"
)
_STANDARD_SPECIAL_LITERALS = ("<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>")
_BOUND_TOKENIZER_FILES = frozenset(
    {
        "binding.json",
        "binding.sha256",
        "tokenizer.json",
        "source-manifest.json",
        "structural-tokens.json",
        "structural-token-ids.json",
        "model-config-65536.json",
    }
)


def validate_contiguous_vocab(vocab: dict[str, int], *, expected_size: int) -> None:
    if type(vocab) is not dict or len(vocab) != expected_size:
        raise ValueError("tokenizer vocabulary size mismatch")
    if any(
        type(token) is not str or type(token_id) is not int
        for token, token_id in vocab.items()
    ):
        raise ValueError("tokenizer vocabulary must map strings to integer IDs")
    ids = set(vocab.values())
    if len(ids) != expected_size or ids != set(range(expected_size)):
        raise ValueError("tokenizer IDs must be unique and contiguous from zero")


class _FrozenTokenizerFacade:
    """Expose only the two non-mutating operations used by the model runtime."""

    __slots__ = ("__tokenizer",)

    def __init__(self, tokenizer: TokenizerType) -> None:
        self.__tokenizer = tokenizer

    def encode_ids(self, text: str) -> tuple[int, ...]:
        return tuple(self.__tokenizer.encode(text, add_special_tokens=False).ids)

    def encode_batch_ids(self, texts: Sequence[str]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(encoding.ids)
            for encoding in self.__tokenizer.encode_batch(
                list(texts), add_special_tokens=False
            )
        )

    def decode(self, ids: Sequence[int]) -> str:
        decoded = self.__tokenizer.decode(list(ids), skip_special_tokens=False)
        if type(decoded) is not str:
            raise ValueError("owned tokenizer returned a non-string decode result")
        return decoded


@dataclass(frozen=True)
class NativeTokenizerBinding:
    bundle_path: Path
    binding_sha256: str
    tokenizer_sha256: str
    vocab_size: int
    structural_token_ids: Mapping[str, int]
    runtime_package: str
    runtime_major_minor: str
    _frozen_tokenizer: _FrozenTokenizerFacade = field(repr=False, compare=False)
    _reserved_pattern: re.Pattern[str] = field(repr=False, compare=False)

    @property
    def structural_token_count(self) -> int:
        return len(self.structural_token_ids)

    @property
    def semantic_record_parser_integrated(self) -> bool:
        return False

    @classmethod
    def load(cls, bundle_path: str | Path = DEFAULT_BUNDLE) -> NativeTokenizerBinding:
        # Recheck the complete owned runtime on every trust-boundary load.  The
        # module-level verification only proves its state at import time.
        _verify_owned_tokenizers_runtime(_TOKENIZERS_RUNTIME_ROOT.resolve())
        runtime_file = getattr(tokenizers, "__file__", None)
        if not isinstance(runtime_file, str):
            raise ValueError("owned tokenizers runtime has no package path")
        runtime_package_root = Path(runtime_file).resolve().parent
        if runtime_package_root != (_TOKENIZERS_RUNTIME_ROOT / "tokenizers").resolve():
            raise ValueError(
                "tokenizers runtime was not loaded from the owned project bundle"
            )
        runtime_binary = runtime_package_root / "tokenizers.pyd"
        if (
            not runtime_binary.is_file()
            or _sha256(runtime_binary) != EXPECTED_RUNTIME_BINARY_SHA256
        ):
            raise ValueError("owned tokenizers runtime binary SHA-256 mismatch")
        root = Path(bundle_path).resolve()
        if not root.is_dir():
            raise ValueError("tokenizer bundle is not a directory")
        entries = tuple(root.iterdir())
        if (
            {entry.name for entry in entries} != _BOUND_TOKENIZER_FILES
            or any(not entry.is_file() or entry.is_symlink() for entry in entries)
        ):
            raise ValueError("tokenizer bundle inventory mismatch")
        binding_path = root / "binding.json"
        actual_binding_hash = _sha256(binding_path)
        if actual_binding_hash != EXPECTED_BINDING_SHA256:
            raise ValueError(
                "binding manifest SHA-256 mismatch: "
                f"expected {EXPECTED_BINDING_SHA256}, got {actual_binding_hash}"
            )
        sidecar = (root / "binding.sha256").read_text(encoding="ascii").strip()
        if sidecar != f"{EXPECTED_BINDING_SHA256}  binding.json":
            raise ValueError("binding SHA-256 sidecar mismatch")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if binding.get("schema_version") != "bosscrafts-native-tokenizer-binding-v1":
            raise ValueError("unsupported tokenizer binding schema")
        if binding.get("lineage") != "bcnlt-native-foundation-65536-v1":
            raise ValueError("unexpected tokenizer lineage")
        expected_scalars = {
            "vocabulary_size": EXPECTED_VOCABULARY_SIZE,
            "minimum_token_id": 0,
            "maximum_token_id": EXPECTED_VOCABULARY_SIZE - 1,
            "tokenizer_json_version": "1.0",
            "tokenizer_model_type": "BPE",
            "token_id_policy": "frozen_exact_no_resize_no_renumber",
        }
        for key, expected in expected_scalars.items():
            if binding.get(key) != expected:
                raise ValueError(f"binding field {key} mismatch")
        runtime = binding.get("runtime")
        if runtime != {
            "package": "tokenizers",
            "required_major_minor": EXPECTED_RUNTIME_MAJOR_MINOR,
        }:
            raise ValueError("tokenizer runtime contract mismatch")
        actual_runtime = ".".join(tokenizers.__version__.split(".")[:2])
        if tokenizers.__version__ != EXPECTED_RUNTIME_VERSION:
            raise ValueError(
                f"tokenizers runtime mismatch: expected {EXPECTED_RUNTIME_VERSION}, "
                f"got {tokenizers.__version__}"
            )

        expected_files = {
            "tokenizer.json",
            "source-manifest.json",
            "structural-tokens.json",
            "structural-token-ids.json",
            "model-config-65536.json",
        }
        file_contracts = binding.get("files")
        if type(file_contracts) is not dict or set(file_contracts) != expected_files:
            raise ValueError("binding file inventory mismatch")
        observed_hashes: dict[str, str] = {}
        for name in sorted(expected_files):
            path = root / name
            contract = file_contracts[name]
            if not path.is_file() or path.stat().st_size != contract.get("bytes"):
                raise ValueError(f"bound file size mismatch: {name}")
            actual_hash = _sha256(path)
            observed_hashes[name] = actual_hash
            if actual_hash != contract.get("sha256"):
                label = (
                    "tokenizer SHA-256 mismatch"
                    if name == "tokenizer.json"
                    else f"bound file SHA-256 mismatch: {name}"
                )
                raise ValueError(label)
        tokenizer_path = root / "tokenizer.json"
        tokenizer_hash = observed_hashes["tokenizer.json"]
        if tokenizer_hash != EXPECTED_TOKENIZER_SHA256:
            raise ValueError("tokenizer SHA-256 mismatch")

        source_manifest = json.loads(
            (root / "source-manifest.json").read_text(encoding="utf-8")
        )
        if (
            source_manifest.get("schema_version")
            != "bcnlt-native-foundation-byte-bpe-v1"
            or source_manifest.get("status") != "complete"
            or source_manifest.get("vocabulary_size") != EXPECTED_VOCABULARY_SIZE
            or source_manifest.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256
            or source_manifest.get("structural_token_count") != 73
        ):
            raise ValueError("source tokenizer manifest contract mismatch")

        raw = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        if raw.get("version") != "1.0" or raw.get("model", {}).get("type") != "BPE":
            raise ValueError("tokenizer JSON version or model type mismatch")
        vocab = raw["model"]["vocab"]
        validate_contiguous_vocab(vocab, expected_size=EXPECTED_VOCABULARY_SIZE)

        structural = json.loads(
            (root / "structural-tokens.json").read_text(encoding="utf-8")
        )
        id_metadata = json.loads(
            (root / "structural-token-ids.json").read_text(encoding="utf-8")
        )
        tokens = structural.get("tokens")
        structural_ids = id_metadata.get("ids")
        if (
            type(tokens) is not list
            or len(tokens) != 73
            or type(structural_ids) is not dict
            or set(structural_ids) != set(tokens)
            or id_metadata.get("token_count") != 73
            or id_metadata.get("id_min") != 4
            or id_metadata.get("id_max") != 76
            or sorted(structural_ids.values()) != list(range(4, 77))
        ):
            raise ValueError("structural token ID metadata mismatch")
        for token, expected_id in structural_ids.items():
            if vocab.get(token) != expected_id:
                raise ValueError(f"structural token renumbered: {token}")

        model_config = json.loads(
            (root / "model-config-65536.json").read_text(encoding="utf-8")
        )
        if (
            model_config.get("vocab_size") != EXPECTED_VOCABULARY_SIZE
            or model_config.get("width") != 1_280
            or model_config.get("parameter_count") != 540_094_726
            or model_config.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256
            or model_config.get("token_id_policy")
            != "frozen_exact_no_resize_no_renumber"
            or model_config.get("tied_input_output_embeddings") is not True
        ):
            raise ValueError("bound model configuration mismatch")

        loaded = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        if loaded.get_vocab_size() != EXPECTED_VOCABULARY_SIZE:
            raise ValueError("loaded tokenizer vocabulary size mismatch")
        structural_items = tuple(structural_ids.items())
        encoded_structural = loaded.encode_batch(
            [token for token, _ in structural_items], add_special_tokens=False
        )
        for (token, expected_id), encoding in zip(
            structural_items, encoded_structural, strict=True
        ):
            if (
                encoding.ids != [expected_id]
                or loaded.token_to_id(token) != expected_id
            ):
                raise ValueError(
                    f"structural token is not atomic at frozen ID: {token}"
                )
        return cls(
            bundle_path=root,
            binding_sha256=actual_binding_hash,
            tokenizer_sha256=tokenizer_hash,
            vocab_size=EXPECTED_VOCABULARY_SIZE,
            structural_token_ids=MappingProxyType(dict(structural_ids)),
            runtime_package="tokenizers",
            runtime_major_minor=actual_runtime,
            _frozen_tokenizer=_FrozenTokenizerFacade(loaded),
            _reserved_pattern=re.compile(
                "|".join(
                    re.escape(literal)
                    for literal in sorted(
                        (*_STANDARD_SPECIAL_LITERALS, *structural_ids),
                        key=len,
                        reverse=True,
                    )
                )
            ),
        )

    def encode(self, text: str) -> list[int]:
        if type(text) is not str:
            raise TypeError("text must be a string")
        ids = list(self._frozen_tokenizer.encode_ids(text))
        if any(
            type(token_id) is not int or not 0 <= token_id < self.vocab_size
            for token_id in ids
        ):
            raise ValueError("tokenizer emitted an out-of-bounds ID")
        return ids

    def encode_payload(self, text: str) -> list[int]:
        """Losslessly encode payload text without emitting reserved control IDs."""

        if type(text) is not str:
            raise TypeError("text must be a string")
        matches = tuple(self._reserved_pattern.finditer(text))
        if not matches:
            ids = self.encode(text)
        else:
            ids = []
            cursor = 0
            for match in matches:
                split = match.start() + 1
                ids.extend(self.encode(text[cursor:split]))
                cursor = split
            ids.extend(self.encode(text[cursor:]))
        if any(0 <= token_id <= 76 for token_id in ids):
            raise ValueError("payload contains reserved tokenizer control text")
        if matches and self.decode(ids) != text:
            raise ValueError("reserved tokenizer literal escape did not round-trip")
        return ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list or tuple")
        if any(type(text) is not str for text in texts):
            raise TypeError("batch texts must be strings")
        rows = [list(ids) for ids in self._frozen_tokenizer.encode_batch_ids(texts)]
        if any(
            type(token_id) is not int or not 0 <= token_id < self.vocab_size
            for ids in rows
            for token_id in ids
        ):
            raise ValueError("tokenizer emitted an out-of-bounds ID")
        return rows

    def encode_payload_batch(self, texts: Sequence[str]) -> list[list[int]]:
        rows = self.encode_batch(texts)
        for index, text in enumerate(texts):
            if self._reserved_pattern.search(text) is not None:
                rows[index] = self.encode_payload(text)
        if any(0 <= token_id <= 76 for ids in rows for token_id in ids):
            raise ValueError("payload contains reserved tokenizer control text")
        return rows

    def decode(self, ids: Sequence[int]) -> str:
        if not isinstance(ids, (list, tuple)):
            raise TypeError("token IDs must be a list or tuple")
        if any(type(token_id) is not int for token_id in ids):
            raise TypeError("token IDs must be integers")
        if any(not 0 <= token_id < self.vocab_size for token_id in ids):
            raise ValueError("token ID out of frozen vocabulary bounds")
        return self._frozen_tokenizer.decode(ids)


def bound_native_65k_config(binding: NativeTokenizerBinding) -> ModelConfig:
    from .config import ModelConfig

    if (
        type(binding) is not NativeTokenizerBinding
        or binding.binding_sha256 != EXPECTED_BINDING_SHA256
        or binding.tokenizer_sha256 != EXPECTED_TOKENIZER_SHA256
        or binding.vocab_size != EXPECTED_VOCABULARY_SIZE
    ):
        raise ValueError("verified native 65K tokenizer binding required")
    return ModelConfig(
        vocab_size=EXPECTED_VOCABULARY_SIZE,
        tokenizer_manifest=binding.binding_sha256,
    )
