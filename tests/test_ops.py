"""Op-level tests with finite-difference grad checks."""

import numpy as np

from tfs.ops import (
    gelu,
    gelu_backward,
    layernorm,
    layernorm_backward,
    softmax,
    softmax_crossentropy,
)


def test_softmax_simplex():
    x = np.random.default_rng(0).standard_normal((3, 5))
    p = softmax(x)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0)
    assert (p >= 0).all()


def test_layernorm_zero_mean_unit_var():
    x = np.random.default_rng(0).standard_normal((4, 6))
    out, _ = layernorm(x, np.ones(6), np.zeros(6))
    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.std(axis=-1), 1.0, atol=1e-2)


def test_layernorm_backward_matches_finite_diff():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5))
    gamma = rng.uniform(0.5, 1.5, size=5)
    beta = rng.standard_normal(5)

    def f(x):
        out, _ = layernorm(x, gamma, beta)
        return out.sum()

    out, cache = layernorm(x, gamma, beta)
    d_out = np.ones_like(out)
    d_x, _, _ = layernorm_backward(d_out, cache)

    eps = 1e-5  # docs/theory.md derives why this is the sweet spot
    probe = x.copy()
    grad_num = np.zeros_like(x)
    for i in range(x.size):
        original = probe.flat[i]
        probe.flat[i] = original + eps
        fp = f(probe)
        probe.flat[i] = original - eps
        fm = f(probe)
        probe.flat[i] = original
        grad_num.flat[i] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(d_x, grad_num, atol=1e-4)


def test_softmax_crossentropy_grad_simplex():
    """Sum of d_logits across vocab axis should be ~0 per example."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 7))
    targets = rng.integers(0, 7, size=(4,))
    _, d = softmax_crossentropy(logits, targets)
    np.testing.assert_allclose(d.sum(axis=-1), 0.0, atol=1e-9)


def test_gelu_backward_matches_finite_diff():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8)
    eps = 1e-5
    fd = (gelu(x + eps) - gelu(x - eps)) / (2 * eps)
    ad = gelu_backward(x, np.ones_like(x))
    np.testing.assert_allclose(ad, fd, atol=1e-4)


def test_softmax_crossentropy_exact_when_confidently_wrong():
    """Loss must be exact even when the target prob underflows.

    log(softmax(x) + 1e-300) saturates at ~690.8 as soon as the target
    probability underflows to 0; the logsumexp form gives the true value.
    """
    logits = np.array([[1000.0, 0.0]])
    targets = np.array([1])
    loss, _ = softmax_crossentropy(logits, targets)
    # exact loss = logsumexp([1000, 0]) - 0 = 1000 + log(1 + e^-1000) = 1000
    assert abs(loss - 1000.0) < 1e-9


def test_softmax_crossentropy_matches_reference_value():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((3, 5))
    targets = np.array([1, 0, 4])
    loss, _ = softmax_crossentropy(logits, targets)
    p = softmax(logits, axis=-1)
    ref = -np.log(p[np.arange(3), targets]).mean()
    assert abs(loss - ref) < 1e-12
