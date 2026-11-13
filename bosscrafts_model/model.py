"""Real neural blocks; stage names are hypotheses, not learned capabilities."""
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig, STAGES

class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return (x * scale.to(x.dtype)) * self.weight

def rotary(x, position_offset=0):
    length, dim = x.shape[-2:]
    inv = 1.0 / (10000 ** (torch.arange(0,dim,2,device=x.device,dtype=torch.float32)/dim))
    positions = torch.arange(
        position_offset,
        position_offset + length,
        device=x.device,
        dtype=torch.float32,
    )
    angle = torch.outer(positions, inv)
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[...,0::2], x[...,1::2]
    return torch.stack((even*cos-odd*sin, even*sin+odd*cos),dim=-1).flatten(-2)

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.norm1, self.norm2 = RMSNorm(c.width), RMSNorm(c.width)
        self.qkv = nn.Linear(c.width,3*c.width,bias=False)
        self.out = nn.Linear(c.width,c.width,bias=False)
        self.gate = nn.Linear(c.width,c.ff_width,bias=False)
        self.up = nn.Linear(c.width,c.ff_width,bias=False)
        self.down = nn.Linear(c.ff_width,c.width,bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        q,k,v = self.qkv(self.norm1(x)).chunk(3,-1)
        def split(t): return t.view(batch,length,self.heads,width//self.heads).transpose(1,2)
        q,k,v = rotary(split(q)),rotary(split(k)),split(v)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
        x = x + self.out(y.transpose(1,2).contiguous().view(batch,length,width))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h))*self.up(h))

    def forward_with_cache(self, x, past_key_value=None):
        batch, length, width = x.shape
        if past_key_value is not None and length != 1:
            raise ValueError("cached block updates require exactly one token")
        q,k,v = self.qkv(self.norm1(x)).chunk(3,-1)
        def split(t): return t.view(batch,length,self.heads,width//self.heads).transpose(1,2)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        q,k,v = rotary(split(q), past_length),rotary(split(k), past_length),split(v)
        if past_key_value is None:
            all_k, all_v = k, v
            y = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
        else:
            past_k, past_v = past_key_value
            if (
                past_k.shape[:-2] != k.shape[:-2]
                or past_v.shape[:-2] != v.shape[:-2]
                or past_k.shape[-1] != k.shape[-1]
                or past_v.shape[-1] != v.shape[-1]
                or past_k.shape[-2] != past_v.shape[-2]
            ):
                raise ValueError("cached key/value tensors are incompatible")
            all_k = torch.cat((past_k, k), dim=-2)
            all_v = torch.cat((past_v, v), dim=-2)
            # The single new query may attend to every cached key and itself;
            # there are no future keys in this incremental update.
            y = F.scaled_dot_product_attention(
                q, all_k, all_v, is_causal=False, dropout_p=0.0
            )
        x = x + self.out(y.transpose(1,2).contiguous().view(batch,length,width))
        h = self.norm2(x)
        x = x + self.down(F.silu(self.gate(h))*self.up(h))
        return x, (all_k, all_v)

class EngineeringModule(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.norm = RMSNorm(c.width)
        self.down = nn.Linear(c.width,c.module_width,bias=False)
        self.up = nn.Linear(c.module_width,c.width,bias=False)

    def forward(self,x):
        return self.up(F.gelu(self.down(self.norm(x))))

class EngineeringModel(nn.Module):
    def __init__(self, c: ModelConfig | None = None):
        super().__init__()
        c = ModelConfig() if c is None else c
        if type(c) is not ModelConfig:
            raise TypeError("c must be a ModelConfig")
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size,c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.final_norm = RMSNorm(c.width)
        if c.modular:
            self.modules_by_stage = nn.ModuleDict({name:EngineeringModule(c) for name in STAGES})
            self.router = nn.Linear(c.width,len(STAGES),bias=True)
        self.gradient_checkpointing = False
        self.apply(self._initialize)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_hidden(self,ids,disabled_stage=None):
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ids must be a batch x sequence int64 tensor")
        if not 0 < ids.shape[1] <= self.config.max_sequence:
            raise ValueError("invalid sequence length")
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("token ID out of configured vocabulary bounds")
        if disabled_stage is not None and disabled_stage not in STAGES:
            raise ValueError("unknown disabled stage")
        x = self.embedding(ids)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        routes = []
        if self.config.modular:
            # Every stage sees the same causal workspace. The next round receives
            # the preceding round's combined contribution, creating a learned loop.
            for _ in range(self.config.module_rounds):
                weights = self.router(x).softmax(-1)
                proposals = []
                for name in STAGES:
                    stage = self.modules_by_stage[name]
                    if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                        proposal = checkpoint(stage, x, use_reentrant=False)
                    else:
                        proposal = stage(x)
                    proposals.append(torch.zeros_like(proposal) if name == disabled_stage else proposal)
                x = x + (torch.stack(proposals,-2)*weights.unsqueeze(-1)).sum(-2)
                routes.append(weights)
        return self.final_norm(x), routes

    def forward(self,ids,disabled_stage=None):
        hidden, routes = self._forward_hidden(ids, disabled_stage=disabled_stage)
        logits = F.linear(hidden,self.embedding.weight) # tied LM head
        return logits, routes

    def forward_target_logits(self, ids, loss_start_index):
        """Project only causal positions that predict target tokens."""
        if type(loss_start_index) is not int:
            raise TypeError("loss_start_index must be an integer")
        if not 0 < loss_start_index < ids.shape[1]:
            raise ValueError("loss_start_index must point inside the sequence")
        hidden, _ = self._forward_hidden(ids)
        return F.linear(
            hidden[:, loss_start_index - 1 : -1], self.embedding.weight
        )

    def forward_prepared(self, prepared, disabled_stage=None):
        """Consume a verified BCNLT pipeline result without retokenizing it."""
        from .bcnlt_source_pipeline import PreparedBCNLTInput

        if type(prepared) is not PreparedBCNLTInput:
            raise TypeError("prepared must be a PreparedBCNLTInput")
        if self.config.tokenizer_manifest != prepared.binding_sha256:
            raise ValueError("prepared input tokenizer binding does not match the model")
        device = self.embedding.weight.device
        ids = torch.tensor((prepared.token_ids,), dtype=torch.long, device=device)
        return self.forward(ids, disabled_stage=disabled_stage)

    def _forward_generation_token(self, ids, past_key_values=None):
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ids must be a batch x sequence int64 tensor")
        if not 0 < ids.shape[1] <= self.config.max_sequence:
            raise ValueError("invalid sequence length")
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("token ID out of configured vocabulary bounds")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("generation cache layer count mismatch")
        x = self.embedding(ids)
        next_key_values = []
        for index, block_module in enumerate(self.blocks):
            block = cast(Block, block_module)
            past = None if past_key_values is None else past_key_values[index]
            x, present = block.forward_with_cache(x, past)
            next_key_values.append(present)
        if self.config.modular:
            for _ in range(self.config.module_rounds):
                weights = self.router(x).softmax(-1)
                proposals = [self.modules_by_stage[name](x) for name in STAGES]
                x = x + (torch.stack(proposals,-2)*weights.unsqueeze(-1)).sum(-2)
        # Generation needs only the newest position. Avoid materializing
        # sequence-length x 65K logits for the initial prompt.
        logits = F.linear(self.final_norm(x[:, -1:]), self.embedding.weight)
        return logits, tuple(next_key_values)

    def generate(self, ids, *, max_new_tokens, eos_token_id=None):
        """Greedy bounded generation with a checkpoint-compatible KV cache."""
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ids must be a batch x sequence int64 tensor")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if ids.shape[1] + max_new_tokens > self.config.max_sequence:
            raise ValueError("generation exceeds the maximum sequence length")
        if eos_token_id is not None and (
            type(eos_token_id) is not int
            or not 0 <= eos_token_id < self.config.vocab_size
        ):
            raise ValueError("eos_token_id is outside the vocabulary")
        generated = ids.clone()
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                logits, past_key_values = self._forward_generation_token(generated)
                for index in range(max_new_tokens):
                    next_token = logits[:, -1].argmax(-1)
                    if eos_token_id is not None:
                        next_token = torch.where(
                            finished,
                            torch.full_like(next_token, eos_token_id),
                            next_token,
                        )
                    generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
                    if eos_token_id is not None:
                        finished |= next_token == eos_token_id
                        if bool(finished.all()):
                            break
                    if index + 1 < max_new_tokens:
                        logits, past_key_values = self._forward_generation_token(
                            next_token.unsqueeze(1), past_key_values
                        )
        finally:
            self.train(was_training)
        return generated

def next_token_loss(logits, ids):
    if ids.shape[1] < 2:
        raise ValueError("at least two tokens required")
    return F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))

def masked_next_token_loss(logits, ids, loss_start_indices):
    """Compute causal LM loss only for packed target tokens."""
    if logits.ndim != 3 or ids.ndim != 2 or logits.shape[:2] != ids.shape:
        raise ValueError("logits and ids must share batch and sequence dimensions")
    if ids.dtype != torch.long or ids.shape[1] < 2:
        raise ValueError("ids must contain at least two int64 tokens")
    if (
        loss_start_indices.ndim != 1
        or loss_start_indices.shape[0] != ids.shape[0]
        or loss_start_indices.dtype != torch.long
        or loss_start_indices.device != ids.device
    ):
        raise ValueError("loss_start_indices must be one int64 value per sequence")
    if bool(
        ((loss_start_indices <= 0) | (loss_start_indices >= ids.shape[1])).any()
    ):
        raise ValueError("loss_start_indices must point inside each packed sequence")
    token_losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(ids.shape[0], ids.shape[1] - 1)
    target_positions = torch.arange(1, ids.shape[1], device=ids.device)
    mask = target_positions.unsqueeze(0) >= loss_start_indices.unsqueeze(1)
    return token_losses[mask].mean()
