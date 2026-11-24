from __future__ import annotations

from typing import Protocol, Sequence


class BaseTokenizer(Protocol):
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError
