"""Multi-head causal self-attention with manual backward pass."""

from __future__ import annotations

import math

import numpy as np

from .ops import Param, softmax


class MultiHeadAttention:
    """Causal multi-head self-attention.

    Input X has shape (B, T, D). Heads operate on D/h per head.
    """

    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator) -> None:
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        bound = math.sqrt(1.0 / d_model)
        self.W_q = Param(rng.uniform(-bound, bound, size=(d_model, d_model)))
        self.W_k = Param(rng.uniform(-bound, bound, size=(d_model, d_model)))
        self.W_v = Param(rng.uniform(-bound, bound, size=(d_model, d_model)))
        self.W_o = Param(rng.uniform(-bound, bound, size=(d_model, d_model)))

    def params(self) -> list[Param]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, D) -> (B, h, T, d_head)."""
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, h, T, d_head) -> (B, T, D)."""
        B, h, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, h * dh)

    def forward(self, X: np.ndarray):
        """Returns (out, cache)."""
        B, T, D = X.shape
        Q = X @ self.W_q.data
        K = X @ self.W_k.data
        V = X @ self.W_v.data
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)
        # scores: (B, h, T, T)
        scores = Qh @ Kh.transpose(0, 1, 3, 2) / math.sqrt(self.d_head)
        # Causal mask: positions can only attend to themselves and earlier.
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
        attn = softmax(scores, axis=-1)
        # out: (B, h, T, d_head)
        ctx = attn @ Vh
        merged = self._merge_heads(ctx)
        out = merged @ self.W_o.data
        cache = (X, Qh, Kh, Vh, attn)
        return out, cache

    def backward(self, d_out: np.ndarray, cache):
        X, Qh, Kh, Vh, attn = cache
        B, T, D = X.shape
        # 1) out = merged @ W_o
        merged = self._merge_heads(attn @ Vh)
        d_W_o = merged.reshape(-1, D).T @ d_out.reshape(-1, D)
        d_merged = d_out @ self.W_o.data.T
        # 2) merged = merge_heads(ctx)
        d_ctx = d_merged.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        # 3) ctx = attn @ Vh
        d_attn = d_ctx @ Vh.transpose(0, 1, 3, 2)
        d_Vh = attn.transpose(0, 1, 3, 2) @ d_ctx
        # 4) attn = softmax(scores). d_scores = attn * (d_attn - sum(d_attn*attn))
        d_scores = attn * (d_attn -
                           (d_attn * attn).sum(axis=-1, keepdims=True))
        # 5) scores = Qh @ Kh.T / sqrt(d_head)
        scale = 1.0 / math.sqrt(self.d_head)
        d_Qh = scale * (d_scores @ Kh)
        d_Kh = scale * (d_scores.transpose(0, 1, 3, 2) @ Qh)
        # 6) Split-heads inverse -> shape (B, T, D)
        d_Q = self._merge_heads(d_Qh)
        d_K = self._merge_heads(d_Kh)
        d_V = self._merge_heads(d_Vh)
        # 7) Q = X @ W_q, etc.
        flat_X = X.reshape(-1, D)
        d_W_q = flat_X.T @ d_Q.reshape(-1, D)
        d_W_k = flat_X.T @ d_K.reshape(-1, D)
        d_W_v = flat_X.T @ d_V.reshape(-1, D)
        d_X = (d_Q @ self.W_q.data.T
               + d_K @ self.W_k.data.T
               + d_V @ self.W_v.data.T)
        # Accumulate into Param grads
        self.W_q.grad = (self.W_q.grad if self.W_q.grad is not None else 0.0) + d_W_q
        self.W_k.grad = (self.W_k.grad if self.W_k.grad is not None else 0.0) + d_W_k
        self.W_v.grad = (self.W_v.grad if self.W_v.grad is not None else 0.0) + d_W_v
        self.W_o.grad = (self.W_o.grad if self.W_o.grad is not None else 0.0) + d_W_o
        return d_X
