from dataclasses import dataclass, asdict
import hashlib
import json
from typing import TypedDict, Unpack

STAGES = ("hypothesis", "prediction", "planning", "construction", "evidence", "revision")


class _ModelConfigOverrides(TypedDict, total=False):
    vocab_size: int
    width: int
    layers: int
    heads: int
    ff_width: int
    module_width: int
    module_rounds: int
    max_sequence: int
    modular: bool
    tokenizer_manifest: str


@dataclass(frozen=True)
class ModelConfig:
    # The production BossCrafts architecture is natively bound to the frozen
    # BCNLT 65K vocabulary. Tiny tests override this explicitly.
    vocab_size: int = 65_536
    width: int = 1280
    layers: int = 24
    heads: int = 20
    ff_width: int = 3072
    module_width: int = 1024
    module_rounds: int = 2
    max_sequence: int = 2048
    modular: bool = True
    tokenizer_manifest: str = "UNBOUND"

    def __post_init__(self):
        for key in ("vocab_size", "width", "layers", "heads", "ff_width",
                    "module_width", "module_rounds", "max_sequence"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.modular) is not bool:
            raise ValueError("modular must be boolean")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide heads with an even head dimension")

    def parameter_count(self):
        d = self.width
        shared = self.vocab_size*d + self.layers*(4*d*d + 3*d*self.ff_width + 2*d) + d
        modular = 6*(2*d*self.module_width+d) + 6*d + 6 if self.modular else 0
        return shared + modular

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def check_manifest(self, manifest):
        if (type(self.tokenizer_manifest) is not str or not self.tokenizer_manifest.strip() or self.tokenizer_manifest == "UNBOUND" or type(manifest) is not str or manifest != self.tokenizer_manifest):
            raise ValueError("No compatible pinned tokenizer manifest; training data rejected")

    @classmethod
    def tiny(cls, **overrides: Unpack[_ModelConfigOverrides]):
        values: _ModelConfigOverrides = dict(
            vocab_size=64,
            width=32,
            layers=2,
            heads=4,
            ff_width=64,
            module_width=16,
            module_rounds=2,
            max_sequence=32,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def baseline(cls):
        return cls(layers=25, modular=False)
