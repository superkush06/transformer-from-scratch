"""Building-block operations with manual forward + backward."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax along `axis`."""
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """LayerNorm over the last axis. Returns (out, cache)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv
    out = gamma * x_hat + beta
    cache = (x_hat, gamma, inv)
    return out, cache


def layernorm_backward(d_out: np.ndarray, cache):
    """Standard LN backward — see Kingma 2016 or ln-bn-bp notes."""
    x_hat, gamma, inv = cache
    N = x_hat.shape[-1]
    d_gamma = (d_out * x_hat).sum(axis=tuple(range(d_out.ndim - 1)))
    d_beta = d_out.sum(axis=tuple(range(d_out.ndim - 1)))
    d_x_hat = d_out * gamma
    d_x = (1.0 / N) * inv * (
        N * d_x_hat
        - d_x_hat.sum(axis=-1, keepdims=True)
        - x_hat * (d_x_hat * x_hat).sum(axis=-1, keepdims=True)
    )
    return d_x, d_gamma, d_beta


def softmax_crossentropy(logits: np.ndarray, targets: np.ndarray):
    """Multi-class CE on the last axis.

    `logits` shape (..., V), `targets` shape (...) with int indices.
    Returns (loss, d_logits).
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    N, V = flat_logits.shape
    # Exact form: log softmax(x)_t = x_t - logsumexp(x). The alternative,
    # log(softmax(x) + eps), silently saturates at -log(eps) once the
    # target probability underflows — i.e. exactly when the model is
    # confidently wrong and the loss value matters most.
    m = flat_logits.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(flat_logits - m).sum(axis=-1)) + m[:, 0]
    log_probs = flat_logits[np.arange(N), flat_targets] - lse
    loss = -log_probs.mean()
    d_logits = softmax(flat_logits, axis=-1)
    d_logits[np.arange(N), flat_targets] -= 1.0
    d_logits /= N
    return float(loss), d_logits.reshape(logits.shape)


def gelu(x: np.ndarray) -> np.ndarray:
    """Approximate GELU (faster than the erf version)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) *
                                    (x + 0.044715 * x ** 3)))


def gelu_backward(x: np.ndarray, d_out: np.ndarray) -> np.ndarray:
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_inner = np.tanh(inner)
    d_inner = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner ** 2) * d_inner
    return d_out * d_gelu


@dataclass
class Param:
    """A parameter tensor with .data, .grad."""
    data: np.ndarray
    grad: np.ndarray = field(default=None)  # type: ignore[assignment]

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)
