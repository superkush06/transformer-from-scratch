"""GPT-style decoder-only transformer."""

from __future__ import annotations

import math

import numpy as np

from .layers import Linear, TransformerBlock
from .ops import (
    Param,
    layernorm,
    layernorm_backward,
    softmax,
    softmax_crossentropy,
)


class GPT:
    """Tiny decoder-only LM with token + position embeddings.

    Hyper-parameters:
      vocab_size: V
      d_model:    embedding dim D
      n_heads:    attention heads
      d_ff:       FFN inner dim
      n_blocks:   number of transformer blocks
      max_T:      maximum sequence length (for position embeddings)
    """

    def __init__(self, *, vocab_size: int, d_model: int, n_heads: int,
                 d_ff: int, n_blocks: int, max_T: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        bound = math.sqrt(1.0 / d_model)
        self.token_emb = Param(rng.uniform(-bound, bound, size=(vocab_size, d_model)))
        self.pos_emb = Param(rng.uniform(-bound, bound, size=(max_T, d_model)))
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, rng)
                       for _ in range(n_blocks)]
        self.ln_f_g = Param(np.ones(d_model))
        self.ln_f_b = Param(np.zeros(d_model))
        self.lm_head = Linear(d_model, vocab_size, bias=False, rng=rng)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_T = max_T

    def params(self) -> list[Param]:
        out = [self.token_emb, self.pos_emb, self.ln_f_g, self.ln_f_b]
        for b in self.blocks:
            out.extend(b.params())
        out.extend(self.lm_head.params())
        return out

    def forward(self, ids: np.ndarray):
        """`ids` shape (B, T). Returns (logits (B, T, V), cache)."""
        B, T = ids.shape
        if T > self.max_T:
            raise ValueError(f"sequence length {T} > max_T {self.max_T}")
        # Fancy indexing would raise a bare IndexError for ids >= V and,
        # worse, silently wrap negative ids to the wrong embedding rows.
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            bad = ids[(ids < 0) | (ids >= self.vocab_size)]
            raise ValueError(
                f"token ids must be in [0, {self.vocab_size}); got {bad[:5].tolist()}")
        x = self.token_emb.data[ids] + self.pos_emb.data[:T][None, :, :]
        block_caches = []
        for blk in self.blocks:
            x, c = blk.forward(x)
            block_caches.append(c)
        h, ln_cache = layernorm(x, self.ln_f_g.data, self.ln_f_b.data)
        logits, lm_cache = self.lm_head.forward(h)
        return logits, (ids, block_caches, ln_cache, lm_cache)

    def loss_and_grads(self, ids: np.ndarray, targets: np.ndarray):
        """Forward + softmax-CE + backward. Returns (loss, ).

        Gradients are accumulated into params' .grad fields (call
        `zero_grad()` first on each).
        """
        logits, cache = self.forward(ids)
        loss, d_logits = softmax_crossentropy(logits, targets)
        self._backward(d_logits, cache)
        return loss

    def _backward(self, d_logits: np.ndarray, cache) -> None:
        ids, block_caches, ln_cache, lm_cache = cache
        # LM head
        d_h = self.lm_head.backward(d_logits, lm_cache)
        # Final LN
        d_x, dg, db = layernorm_backward(d_h, ln_cache)
        self.ln_f_g.grad = (self.ln_f_g.grad if self.ln_f_g.grad is not None else 0.0) + dg
        self.ln_f_b.grad = (self.ln_f_b.grad if self.ln_f_b.grad is not None else 0.0) + db
        # Blocks (reverse order)
        for blk, c in zip(reversed(self.blocks), reversed(block_caches), strict=False):
            d_x = blk.backward(d_x, c)
        # Position emb: gradient sums over batch
        d_pos = d_x.sum(axis=0)
        T = d_pos.shape[0]
        self.pos_emb.grad = (self.pos_emb.grad if self.pos_emb.grad is not None
                             else np.zeros_like(self.pos_emb.data))
        self.pos_emb.grad[:T] += d_pos
        # Token emb: scatter-add by index
        self.token_emb.grad = (self.token_emb.grad if self.token_emb.grad is not None
                               else np.zeros_like(self.token_emb.data))
        flat_ids = ids.reshape(-1)
        flat_d = d_x.reshape(-1, self.d_model)
        np.add.at(self.token_emb.grad, flat_ids, flat_d)

    @np.errstate(all="ignore")
    def generate(self, prompt: np.ndarray, max_new: int,
                 temperature: float = 1.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """Autoregressive sampling. `prompt` shape (T,)."""
        rng = rng or np.random.default_rng(0)
        ids = list(map(int, prompt))
        for _ in range(max_new):
            ctx = np.array(ids[-self.max_T:])[None, :]
            logits, _ = self.forward(ctx)
            last = logits[0, -1] / max(1e-6, temperature)
            probs = softmax(last)
            ids.append(int(rng.choice(self.vocab_size, p=probs)))
        return np.array(ids)
