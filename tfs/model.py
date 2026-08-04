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

    def named_params(self) -> list[tuple[str, Param]]:
        """Every parameter tensor, with a dotted name.

        `params()` is exactly the values of this list, in this order — so a
        gradient check can iterate the model instead of hard-coding a
        parallel list that silently rots when a Param is added.
        """
        out = [("token_emb", self.token_emb), ("pos_emb", self.pos_emb),
               ("ln_f_g", self.ln_f_g), ("ln_f_b", self.ln_f_b)]
        for i, b in enumerate(self.blocks):
            out += [(f"blocks.{i}.{n}", p) for n, p in b.named_params()]
        out += [(f"lm_head.{n}", p) for n, p in self.lm_head.named_params()]
        return out

    def params(self) -> list[Param]:
        return [p for _, p in self.named_params()]

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

    def forward_step(self, ids_new: np.ndarray, pos: int, kv):
        """One cached decoding step.

        `ids_new` is (B, 1) — the single new token. `pos` is its absolute
        position (so it picks up the right row of the position table), and
        `kv` is the per-block (K, V) cache covering positions [0, pos).
        Returns (logits (B, V) for the new token, extended cache).
        """
        if ids_new.min() < 0 or ids_new.max() >= self.vocab_size:
            raise ValueError(f"token ids must be in [0, {self.vocab_size})")
        if pos >= self.max_T:
            raise ValueError(f"position {pos} >= max_T {self.max_T}")
        x = self.token_emb.data[ids_new] + self.pos_emb.data[pos][None, None, :]
        new_kv = []
        for blk, blk_kv in zip(self.blocks, kv, strict=True):
            x, blk_kv = blk.forward_step(x, blk_kv)
            new_kv.append(blk_kv)
        h, _ = layernorm(x, self.ln_f_g.data, self.ln_f_b.data)
        logits, _ = self.lm_head.forward(h)
        return logits[:, -1, :], new_kv

    def _sample(self, logits: np.ndarray, temperature: float,
                rng: np.random.Generator) -> int:
        if temperature == 0.0:
            return int(np.argmax(logits))
        return int(rng.choice(self.vocab_size, p=softmax(logits / temperature)))

    def generate(self, prompt: np.ndarray, max_new: int,
                 temperature: float = 1.0,
                 rng: np.random.Generator | None = None,
                 use_cache: bool = True) -> np.ndarray:
        """Autoregressive decoding. `prompt` has shape (T,).

        temperature > 0 samples from softmax(logits / temperature);
        temperature == 0 is exact greedy argmax decoding. Negative
        temperatures are rejected rather than silently clamped.

        With `use_cache=True` each step feeds only the new token and reuses
        the cached keys and values, which is bit-for-bit the same arithmetic
        as re-running the whole prefix (`tests/test_kv_cache.py` pins the
        equivalence). The cache is rebuilt whenever the context window has
        to slide past `max_T`: the learned *absolute* position table means
        every surviving token changes its position id, so its keys and
        values are no longer valid. That is a real cost of absolute
        positions, and one of the reasons relative schemes exist.
        """
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0; got {temperature}")
        rng = rng or np.random.default_rng(0)
        ids = list(map(int, prompt))
        if not use_cache:
            for _ in range(max_new):
                ctx = np.array(ids[-self.max_T:])[None, :]
                logits, _ = self.forward(ctx)
                ids.append(self._sample(logits[0, -1], temperature, rng))
            return np.array(ids)

        kv = None
        n_cached = 0
        for _ in range(max_new):
            ctx = ids[-self.max_T:]
            if kv is None or n_cached != len(ctx) - 1:
                # Prefill, or re-fill after the window slid: run the full
                # forward over the window and keep the keys/values it built.
                logits, cache = self.forward(np.array(ctx)[None, :])
                kv = [TransformerBlock.kv_from_cache(c) for c in cache[1]]
                n_cached = len(ctx)
                last = logits[0, -1]
            else:
                last, kv = self.forward_step(np.array([[ctx[-1]]]), n_cached, kv)
                n_cached += 1
                last = last[0]
            ids.append(self._sample(last, temperature, rng))
        return np.array(ids)
