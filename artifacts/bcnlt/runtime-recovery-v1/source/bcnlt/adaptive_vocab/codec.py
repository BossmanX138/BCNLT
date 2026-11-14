from __future__ import annotations

from typing import Sequence

from .manifest import AdaptiveVocabularyManifest


class MacroCodec:
    def __init__(self, manifest: AdaptiveVocabularyManifest) -> None:
        self.manifest = manifest
        self._macro_by_id = {macro.id: macro for macro in manifest.macros}
        self._trie: dict[int | None, object] = {}
        for macro in manifest.macros:
            node = self._trie
            for token_id in macro.expansion:
                node = node.setdefault(token_id, {})  # type: ignore[assignment]
            matches = node.setdefault(None, [])  # type: ignore[union-attr]
            matches.append(macro)  # type: ignore[union-attr]

    def substitute(self, base_ids: Sequence[int], field: str) -> list[int]:
        if any(
            not isinstance(token_id, int)
            or token_id < 0
            or token_id >= self.manifest.base_vocab_size
            for token_id in base_ids
        ):
            raise ValueError("macro substitution accepts base-token IDs only")
        result: list[int] = []
        cursor = 0
        while cursor < len(base_ids):
            node = self._trie
            probe = cursor
            matches = []
            while probe < len(base_ids) and base_ids[probe] in node:
                node = node[base_ids[probe]]  # type: ignore[assignment,index]
                probe += 1
                matches.extend(
                    macro
                    for macro in node.get(None, [])  # type: ignore[union-attr]
                    if field in macro.allowed_fields
                )
            if matches:
                chosen = min(matches, key=lambda macro: (-len(macro.expansion), macro.id))
                result.append(chosen.id)
                cursor += len(chosen.expansion)
            else:
                result.append(base_ids[cursor])
                cursor += 1
        return result

    def expand(self, ids: Sequence[int]) -> list[int]:
        result: list[int] = []
        for token_id in ids:
            macro = self._macro_by_id.get(token_id)
            if macro is not None:
                result.extend(macro.expansion)
            elif 0 <= token_id < self.manifest.v1_vocab_size:
                result.append(token_id)
            else:
                raise ValueError(f"token id is outside the adaptive vocabulary: {token_id}")
        return result
