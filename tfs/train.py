"""Tiny SGD/Adam training loop with manual backward."""

from __future__ import annotations

import numpy as np


class AdamLite:
    """Adam optimiser over a flat list of Param-like objects (with .data, .grad)."""

    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8) -> None:
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self._t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self._m[i] = self.b1 * self._m[i] + (1 - self.b1) * p.grad
            self._v[i] = self.b2 * self._v[i] + (1 - self.b2) * (p.grad ** 2)
            m_hat = self._m[i] / (1 - self.b1 ** self._t)
            v_hat = self._v[i] / (1 - self.b2 ** self._t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
