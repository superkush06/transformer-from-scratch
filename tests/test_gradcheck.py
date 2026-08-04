"""Finite-difference gradient check for EVERY parameter tensor.

A 2-block model has 29 parameter tensors. The op-level tests only pin
d_X (attention) and d_x (LayerNorm); a sign error in d_W_k, a dropped
LN gamma accumulation, or a broken embedding scatter-add would sail
through them. This module checks each tensor against central
differences. Token ids contain repeats on purpose to stress the
np.add.at scatter-add path.

The parametrisation walks `GPT.named_params()`, so a Param added to the
model is grad-checked the day it is added — there is no parallel list
here to forget to update.
"""

import numpy as np
import pytest

from tfs.model import GPT
from tfs.ops import softmax_crossentropy

# Repeated ids in both rows: the scatter-add must accumulate, not overwrite.
IDS = np.array([[1, 2, 1, 2, 5], [3, 3, 0, 1, 1]])
TGT = np.array([[2, 1, 2, 5, 6], [3, 0, 1, 1, 4]])


def build() -> GPT:
    return GPT(vocab_size=7, d_model=8, n_heads=2, d_ff=16,
               n_blocks=2, max_T=6, seed=0)


NAMES = [name for name, _ in build().named_params()]


def test_named_params_covers_every_parameter():
    """`params()` and `named_params()` must not drift apart."""
    model = build()
    assert len(NAMES) == 29
    assert len(set(NAMES)) == 29
    assert [id(p) for _, p in model.named_params()] == [id(p) for p in model.params()]


@pytest.mark.parametrize("name", NAMES)
def test_param_grad_matches_finite_differences(name):
    model = build()
    for p in model.params():
        p.zero_grad()
    model.loss_and_grads(IDS, TGT)
    param = dict(model.named_params())[name]
    analytic = np.asarray(param.grad)

    def loss() -> float:
        logits, _ = model.forward(IDS)
        value, _ = softmax_crossentropy(logits, TGT)
        return value

    n = param.data.size
    if n <= 40:
        coords = np.arange(n)
    else:  # sample coordinates for the big tensors, deterministic per name
        rng = np.random.default_rng(sum(name.encode()))
        coords = rng.choice(n, size=25, replace=False)

    eps = 1e-5  # docs/theory.md derives why this is the sweet spot
    for c in coords:
        orig = param.data.flat[c]
        param.data.flat[c] = orig + eps
        lp = loss()
        param.data.flat[c] = orig - eps
        lm = loss()
        param.data.flat[c] = orig
        fd = (lp - lm) / (2 * eps)
        ad = analytic.flat[c]
        assert abs(ad - fd) <= 1e-7 + 1e-4 * abs(fd), (
            f"{name}[{c}]: analytic={ad:.3e} fd={fd:.3e}")
