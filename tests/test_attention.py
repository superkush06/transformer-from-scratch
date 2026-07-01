"""Multi-head attention shape + grad-check tests."""

import numpy as np

from tfs.attention import MultiHeadAttention


def test_mha_output_shape():
    rng = np.random.default_rng(0)
    mha = MultiHeadAttention(d_model=16, n_heads=4, rng=rng)
    X = rng.standard_normal((2, 5, 16))
    out, _ = mha.forward(X)
    assert out.shape == (2, 5, 16)


def test_mha_causal_mask_blocks_future():
    """Position t should not depend on position t+1 (deterministic check)."""
    rng = np.random.default_rng(0)
    mha = MultiHeadAttention(d_model=8, n_heads=2, rng=rng)
    X = rng.standard_normal((1, 4, 8))
    out_full, _ = mha.forward(X)
    # Perturb the last position; first three positions should be unchanged.
    X2 = X.copy()
    X2[0, -1] += 1.0
    out_pert, _ = mha.forward(X2)
    np.testing.assert_allclose(out_full[0, :3], out_pert[0, :3], atol=1e-8)


def test_mha_backward_finite_diff_input():
    """Gradient wrt X matches finite differences."""
    rng = np.random.default_rng(0)
    mha = MultiHeadAttention(d_model=8, n_heads=2, rng=rng)
    X = rng.standard_normal((1, 3, 8))

    def loss_of(X):
        out, _ = mha.forward(X)
        return float(out.sum())

    out, cache = mha.forward(X)
    d_out = np.ones_like(out)
    d_X = mha.backward(d_out, cache)

    eps = 1e-5
    grad_num = np.zeros_like(X)
    for i in range(X.size):
        flat = X.flatten()
        flat[i] += eps; fp = loss_of(flat.reshape(X.shape))
        flat[i] -= 2*eps; fm = loss_of(flat.reshape(X.shape))
        flat[i] += eps
        grad_num.flat[i] = (fp - fm) / (2*eps)
    np.testing.assert_allclose(d_X, grad_num, atol=1e-3)
