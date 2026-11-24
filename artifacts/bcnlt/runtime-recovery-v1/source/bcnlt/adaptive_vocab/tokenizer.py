from __future__ import annotations

from collections.abc import Iterable, Sequence

from bcnlt.semantic_record import SemanticTokenRecord
from bcnlt.tokenizer import BCNLTTokenizer

from .codec import MacroCodec
from .manifest import AdaptiveVocabularyManifest, validate_adaptive_manifest


class AdaptiveBCNLTTokenizer:
    def __init__(
        self, v1: BCNLTTokenizer, manifest: AdaptiveVocabularyManifest
    ) -> None:
        validate_adaptive_manifest(manifest, v1.manifest)
        self.v1 = v1
        self.base = v1.base
        self.manifest = manifest
        self.codec = MacroCodec(manifest)

    def encode_text(self, text: str) -> list[int]:
        return self.v1.encode_text(text)

    def encode_raw_code(self, source: str) -> list[int]:
        return self.v1.encode_raw_code(source)

    def encode_records(self, records: Iterable[SemanticTokenRecord]) -> list[int]:
        return self._substitute_stream(self.v1.encode_records(records))

    def encode_source(
        self,
        source: str,
        path: str = "<memory>",
        *,
        language: str = "python",
        dialect: str | None = None,
    ) -> list[int]:
        return self._substitute_stream(
            self.v1.encode_source(
                source,
                path=path,
                language=language,
                dialect=dialect,
            )
        )

    def decode(self, ids: Sequence[int]) -> str:
        return self.v1.decode(self.codec.expand(ids))

    def decode_records(self, ids: Sequence[int]) -> tuple[SemanticTokenRecord, ...]:
        return self.v1.decode_records(self.codec.expand(ids))

    def _substitute_stream(self, ids: Sequence[int]) -> list[int]:
        result: list[int] = []
        field: str | None = None
        cursor = 0
        while cursor < len(ids):
            token_id = ids[cursor]
            if token_id < self.manifest.base_vocab_size:
                end = cursor + 1
                while end < len(ids) and ids[end] < self.manifest.base_vocab_size:
                    end += 1
                run = ids[cursor:end]
                result.extend(
                    self.codec.substitute(run, field) if field is not None else run
                )
                cursor = end
                continue
            result.append(token_id)
            token = self.v1.manifest.id_to_token.get(token_id)
            if token is None:
                raise ValueError(f"v1 semantic stream contains unknown structural id: {token_id}")
            if token.startswith("<|bcnlt:field:") and token.endswith(":begin|>"):
                field = token.removeprefix("<|bcnlt:field:").removesuffix(":begin|>")
            elif token.startswith("<|bcnlt:field:") and token.endswith(":end|>"):
                field = None
            cursor += 1
        return result
