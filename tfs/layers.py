"""Linear, FFN, and transformer block."""

from __future__ import annotations

import math

import numpy as np

from .attention import MultiHeadAttention
from .ops import (
    Param,
    gelu,
    gelu_backward,
    layernorm,
    layernorm_backward,
)


class Linear:
    """y = x @ W + b. b is optional."""

    def __init__(self, in_dim: int, out_dim: int, *, bias: bool,
                 rng: np.random.Generator) -> None:
        bound = math.sqrt(1.0 / in_dim)
        self.W = Param(rng.uniform(-bound, bound, size=(in_dim, out_dim)))
        self.b = Param(np.zeros(out_dim)) if bias else None

    def params(self) -> list[Param]:
        return [self.W] + ([self.b] if self.b is not None else [])

    def forward(self, x: np.ndarray):
        y = x @ self.W.data
        if self.b is not None:
            y = y + self.b.data
        return y, x

    def backward(self, d_y: np.ndarray, cache):
        x = cache
        d_W = x.reshape(-1, x.shape[-1]).T @ d_y.reshape(-1, d_y.shape[-1])
        self.W.grad = (self.W.grad if self.W.grad is not None else 0.0) + d_W
        if self.b is not None:
            d_b = d_y.reshape(-1, d_y.shape[-1]).sum(axis=0)
            self.b.grad = (self.b.grad if self.b.grad is not None else 0.0) + d_b
        d_x = d_y @ self.W.data.T
        return d_x


class FFN:
    """Two-layer position-wise feed-forward with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator) -> None:
        self.up = Linear(d_model, d_ff, bias=True, rng=rng)
        self.down = Linear(d_ff, d_model, bias=True, rng=rng)

    def params(self) -> list[Param]:
        return self.up.params() + self.down.params()

    def forward(self, x: np.ndarray):
        u, u_cache = self.up.forward(x)
        a = gelu(u)
        out, d_cache = self.down.forward(a)
        return out, (u, u_cache, d_cache)

    def backward(self, d_out, cache):
        u, u_cache, d_cache = cache
        d_a = self.down.backward(d_out, d_cache)
        d_u = gelu_backward(u, d_a)
        d_x = self.up.backward(d_u, u_cache)
        return d_x


class TransformerBlock:
    """Pre-norm transformer block: x -> x + Attn(LN(x)); x -> x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 rng: np.random.Generator) -> None:
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ffn = FFN(d_model, d_ff, rng)
        self.ln1_g = Param(np.ones(d_model))
        self.ln1_b = Param(np.zeros(d_model))
        self.ln2_g = Param(np.ones(d_model))
        self.ln2_b = Param(np.zeros(d_model))

    def params(self) -> list[Param]:
        return (self.attn.params() + self.ffn.params()
                + [self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b])

    def forward(self, x: np.ndarray):
        h1, c1 = layernorm(x, self.ln1_g.data, self.ln1_b.data)
        a, attn_cache = self.attn.forward(h1)
        x1 = x + a
        h2, c2 = layernorm(x1, self.ln2_g.data, self.ln2_b.data)
        f, ffn_cache = self.ffn.forward(h2)
        x2 = x1 + f
        return x2, (c1, attn_cache, c2, ffn_cache)

    def backward(self, d_out, cache):
        c1, attn_cache, c2, ffn_cache = cache
        # x2 = x1 + f
        d_f = d_out
        d_x1_from_ffn = d_out
        d_h2 = self.ffn.backward(d_f, ffn_cache)
        d_x1_from_ln2, dg2, db2 = layernorm_backward(d_h2, c2)
        self.ln2_g.grad = (self.ln2_g.grad if self.ln2_g.grad is not None else 0.0) + dg2
        self.ln2_b.grad = (self.ln2_b.grad if self.ln2_b.grad is not None else 0.0) + db2
        d_x1 = d_x1_from_ffn + d_x1_from_ln2
        # x1 = x + a
        d_a = d_x1
        d_x_from_attn = d_x1
        d_h1 = self.attn.backward(d_a, attn_cache)
        d_x_from_ln1, dg1, db1 = layernorm_backward(d_h1, c1)
        self.ln1_g.grad = (self.ln1_g.grad if self.ln1_g.grad is not None else 0.0) + dg1
        self.ln1_b.grad = (self.ln1_b.grad if self.ln1_b.grad is not None else 0.0) + db1
        d_x = d_x_from_attn + d_x_from_ln1
        return d_x
