from __future__ import annotations

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def write_byte_level_tokenizer(path: Path, corpus: str = "hello world cafe code x = 1") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    training_text = path / "training.txt"
    training_text.write_text(corpus, encoding="utf-8")
    byte_level = ByteLevelBPETokenizer()
    byte_level.train(
        files=[str(training_text)],
        vocab_size=300,
        min_frequency=1,
        special_tokens=["<unk>"],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=byte_level._tokenizer,
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)
    training_text.unlink()
    return path
