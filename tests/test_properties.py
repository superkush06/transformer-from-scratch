"""Randomised property tests: laws that hold for every valid input.

The rest of the suite pins fixed cases — this module pins the *algebra*.
Each test draws many inputs from a seeded generator and asserts an
identity that the mathematics guarantees, so a change that happens to
keep the fixtures passing still has to keep the law.

Seeds are fixed, so a failure here is reproducible from the test name
alone. Where a law only holds exactly at `eps = 0` (LayerNorm's
normalisation is exact only without the variance floor), the test says so
and passes `eps=0.0` rather than loosening the tolerance until it passes.
"""

import math

import numpy as np
import pytest

from tfs.attention import MultiHeadAttention
from tfs.layers import TransformerBlock
from tfs.model import GPT
from tfs.ops import (
    layernorm,
    layernorm_backward,
    softmax,
    softmax_crossentropy,
)
from tfs.train import AdamLite

DRAWS = 200

# ln of the largest float64: np.exp(x) overflows to inf above this, which is
# the regime the max-subtraction inside tfs.ops.softmax exists to survive.
EXP_OVERFLOW = float(np.log(np.finfo(np.float64).max))


def shapes(rng, n=DRAWS, max_rows=6, max_cols=12, log10_max_scale=2):
    """Random (rows, cols) pairs, plus a wildly varying logit scale."""
    for _ in range(n):
        yield (int(rng.integers(1, max_rows + 1)),
               int(rng.integers(2, max_cols + 1)),
               float(10.0 ** rng.uniform(-3, log10_max_scale)))


# --------------------------------------------------------------------------
# softmax
# --------------------------------------------------------------------------

def test_softmax_lands_on_the_probability_simplex():
    """Every row is non-negative and sums to one.

    softmax is defined as a normalisation, so this must hold for any finite
    input — including logits large enough that a naive exp() overflows,
    which is the case the max-subtraction in tfs.ops.softmax exists for.
    The scale is drawn up to 1e3 precisely so some draws land above
    EXP_OVERFLOW; the count is asserted, because a version of this test
    that never reaches that regime would pass with the max-subtraction
    deleted.
    """
    rng = np.random.default_rng(20260727)
    worst = 0.0
    overflowing = 0
    for rows, cols, scale in shapes(rng, log10_max_scale=3):
        x = rng.standard_normal((rows, cols)) * scale
        overflowing += int(x.max() > EXP_OVERFLOW)
        p = softmax(x)
        assert (p >= 0.0).all() and (p <= 1.0).all()
        worst = max(worst, float(np.abs(p.sum(-1) - 1.0).max()))
    assert overflowing > 0, "no draw reached the naive-exp overflow regime"
    assert worst < 1e-14, worst


def test_softmax_is_invariant_to_a_constant_shift():
    """softmax(x + c) == softmax(x) for any scalar c.

    The identity is exact in real arithmetic because the constant cancels
    between numerator and denominator. In floating point it is the reason
    subtracting the row max is free: the answer does not change, only the
    dynamic range does.
    """
    rng = np.random.default_rng(20260728)
    worst = 0.0
    overflowing = 0
    for rows, cols, scale in shapes(rng, log10_max_scale=3):
        x = rng.standard_normal((rows, cols)) * scale
        c = rng.uniform(-500, 500)
        overflowing += int((x + c).max() > EXP_OVERFLOW)
        gap = np.abs(softmax(x + c) - softmax(x))
        # max() would quietly swallow a NaN row, which is exactly what an
        # overflowing exp() produces; check finiteness before reducing.
        assert np.isfinite(gap).all()
        worst = max(worst, float(gap.max()))
    assert overflowing > 0, "no shifted draw reached the overflow regime"
    assert worst < 1e-12, worst


def test_softmax_jacobian_matches_diag_p_minus_outer_p():
    """J = diag(p) - p pᵀ, measured against central differences.

    Every attention backward in the repo is a contraction with this
    Jacobian (`d_scores = attn * (d_attn - sum(d_attn * attn))`), so the
    closed form deserves to be checked directly rather than only through
    the layers that use it.
    """
    rng = np.random.default_rng(20260729)
    worst = 0.0
    for _ in range(60):
        n = int(rng.integers(2, 9))
        x = rng.standard_normal(n) * rng.uniform(0.2, 3.0)
        p = softmax(x)
        closed = np.diag(p) - np.outer(p, p)
        eps = 1e-5
        numeric = np.zeros((n, n))
        for j in range(n):
            up, down = x.copy(), x.copy()
            up[j] += eps
            down[j] -= eps
            numeric[:, j] = (softmax(up) - softmax(down)) / (2 * eps)
        worst = max(worst, float(np.abs(closed - numeric).max()))
    assert worst < 1e-9, worst


def test_temperature_sharpens_monotonically():
    """Lowering the temperature never lowers the top probability.

    softmax(x / T) is a monotone family: as T falls the distribution moves
    toward the argmax one-hot, so max_i p_i is non-increasing in T. This is
    what makes `temperature=0` the limit of the sampling path rather than a
    special case bolted on beside it.
    """
    rng = np.random.default_rng(20260730)
    temps = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 8.0])
    for _ in range(DRAWS):
        x = rng.standard_normal(int(rng.integers(2, 20))) * rng.uniform(0.1, 5)
        peaks = [float(softmax(x / t).max()) for t in temps]
        assert all(a >= b - 1e-12 for a, b in zip(peaks, peaks[1:], strict=False))


# --------------------------------------------------------------------------
# LayerNorm
# --------------------------------------------------------------------------

def test_layernorm_output_moments_match_the_closed_form():
    """Row mean is 0; row variance is var/(var + eps), not exactly 1.

    The variance floor is not cosmetic — it is why the output variance is
    measurably below one, and quoting "unit variance" without it is the
    kind of claim this test exists to keep honest.
    """
    rng = np.random.default_rng(20260731)
    worst_mean = worst_var = 0.0
    for rows, cols, scale in shapes(rng, max_cols=16):
        if cols < 3:
            continue
        x = rng.standard_normal((rows, cols)) * scale
        eps = 1e-5
        out, _ = layernorm(x, np.ones(cols), np.zeros(cols), eps=eps)
        var = x.var(axis=-1)
        worst_mean = max(worst_mean, float(np.abs(out.mean(-1)).max()))
        predicted = var / (var + eps)
        worst_var = max(worst_var,
                        float(np.abs(out.var(-1) - predicted).max()))
    assert worst_mean < 1e-12, worst_mean
    assert worst_var < 1e-11, worst_var


def test_layernorm_is_invariant_to_affine_rescaling_of_its_input():
    """LN(a·x + b) == LN(x) for a > 0, exactly, when eps = 0.

    Centring removes b and dividing by the standard deviation removes a.
    That invariance is the whole point of the layer, and it is exact only
    without the variance floor — with eps > 0 it holds to O(eps/var).
    """
    rng = np.random.default_rng(20260801)
    worst = 0.0
    for rows, cols, _ in shapes(rng, max_cols=16):
        if cols < 3:
            continue
        x = rng.standard_normal((rows, cols))
        a, b = rng.uniform(0.1, 10.0), rng.uniform(-20, 20)
        g, z = np.ones(cols), np.zeros(cols)
        base, _ = layernorm(x, g, z, eps=0.0)
        scaled, _ = layernorm(a * x + b, g, z, eps=0.0)
        worst = max(worst, float(np.abs(base - scaled).max()))
    assert worst < 1e-11, worst


def test_layernorm_backward_is_orthogonal_to_both_normalisation_directions():
    """sum_j d_x_j == 0 and sum_j d_x_j·x̂_j == 0.

    LayerNorm's output is unchanged by shifting its input (that is the
    centring) and, at eps = 0, by scaling it (that is the division by the
    standard deviation). A directional derivative along a direction the
    output cannot see must vanish, so d_x is orthogonal to both 1 and x̂.
    Getting either wrong is the classic LayerNorm-backward bug: it leaves
    a gradient that pushes on a degree of freedom the layer discards.
    """
    rng = np.random.default_rng(20260802)
    worst_shift = worst_scale = 0.0
    for rows, cols, _ in shapes(rng, n=120, max_cols=16):
        if cols < 3:
            continue
        x = rng.standard_normal((rows, cols)) * rng.uniform(0.2, 4.0)
        gamma = rng.uniform(0.3, 2.0, size=cols)
        beta = rng.standard_normal(cols)
        _, cache = layernorm(x, gamma, beta, eps=0.0)
        d_out = rng.standard_normal((rows, cols))
        d_x, _, _ = layernorm_backward(d_out, cache)
        x_hat = cache[0]
        scale = max(1.0, float(np.abs(d_x).max()))
        worst_shift = max(worst_shift,
                          float(np.abs(d_x.sum(-1)).max()) / scale)
        worst_scale = max(worst_scale,
                          float(np.abs((d_x * x_hat).sum(-1)).max()) / scale)
    assert worst_shift < 1e-12, worst_shift
    assert worst_scale < 1e-12, worst_scale


# --------------------------------------------------------------------------
# cross-entropy
# --------------------------------------------------------------------------

def test_cross_entropy_is_shift_invariant_and_never_negative():
    """L(x + c) == L(x) >= 0, and L == ln V exactly at uniform logits.

    Cross-entropy is a function of the softmax, so it inherits softmax's
    shift invariance; it is a negative log probability, so it cannot be
    negative; and a uniform distribution over V classes has surprisal
    ln V for whichever class turns up.
    """
    rng = np.random.default_rng(20260803)
    worst_shift = 0.0
    for _ in range(DRAWS):
        rows = int(rng.integers(1, 8))
        V = int(rng.integers(2, 20))
        logits = rng.standard_normal((rows, V)) * rng.uniform(0.1, 10)
        targets = rng.integers(0, V, size=rows)
        base, _ = softmax_crossentropy(logits, targets)
        shifted, _ = softmax_crossentropy(logits + rng.uniform(-300, 300),
                                          targets)
        assert base >= 0.0
        worst_shift = max(worst_shift, abs(base - shifted))
        flat, _ = softmax_crossentropy(np.zeros((rows, V)), targets)
        assert abs(flat - math.log(V)) < 1e-12
    assert worst_shift < 1e-11, worst_shift


def test_cross_entropy_gradient_is_a_zero_sum_probability_residual():
    """d_logits sums to zero over the vocabulary, and equals (p - onehot)/N.

    The gradient of -log p_t with respect to the logits is p - e_t; the
    mean over N predictions divides by N. Both facts are checked, and the
    zero-sum property is what makes the loss blind to a constant shift.
    """
    rng = np.random.default_rng(20260804)
    worst_sum = worst_form = 0.0
    for _ in range(DRAWS):
        rows = int(rng.integers(1, 8))
        V = int(rng.integers(2, 20))
        logits = rng.standard_normal((rows, V)) * rng.uniform(0.1, 5)
        targets = rng.integers(0, V, size=rows)
        _, d = softmax_crossentropy(logits, targets)
        expected = softmax(logits, axis=-1)
        expected[np.arange(rows), targets] -= 1.0
        expected /= rows
        worst_sum = max(worst_sum, float(np.abs(d.sum(-1)).max()))
        worst_form = max(worst_form, float(np.abs(d - expected).max()))
    assert worst_sum < 1e-15, worst_sum
    assert worst_form == 0.0, worst_form


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------

def test_attention_output_is_a_convex_combination_of_its_values():
    """Every context vector lies inside the box spanned by the values it may see.

    Attention weights are non-negative and sum to one, so the context at
    position i is a convex combination of V rows 0..i — and therefore
    bounded coordinate-wise by their min and max. A mask that leaked, or a
    softmax that did not normalise, would put a coordinate outside the box.
    """
    rng = np.random.default_rng(20260805)
    for _ in range(40):
        heads = int(rng.integers(1, 5))
        d_head = int(rng.integers(2, 6))
        D = heads * d_head
        T = int(rng.integers(1, 9))
        mha = MultiHeadAttention(d_model=D, n_heads=heads, rng=rng)
        X = rng.standard_normal((2, T, D)) * rng.uniform(0.2, 3.0)
        out, cache = mha.forward(X)
        _, _, _, Vh, attn = cache
        # rows of attn are probabilities
        np.testing.assert_allclose(attn.sum(-1), 1.0, atol=1e-13)
        assert (attn >= 0).all()
        # and nothing above the diagonal carries weight
        assert float(np.triu(attn, k=1).max()) == 0.0
        ctx = attn @ Vh
        for i in range(T):
            lo = Vh[:, :, :i + 1, :].min(axis=2)
            hi = Vh[:, :, :i + 1, :].max(axis=2)
            assert (ctx[:, :, i, :] >= lo - 1e-12).all()
            assert (ctx[:, :, i, :] <= hi + 1e-12).all()
        assert out.shape == X.shape


def test_no_future_token_can_move_an_earlier_logit():
    """Rewriting the suffix leaves the prefix logits bit-identical.

    Causality is not "approximately enforced by a large negative number";
    -1e9 underflows to exactly zero weight in float64, so the deviation
    below is 0.0 and not merely small. This is the ablation form of the
    claim: intervene on the future, observe the past.
    """
    rng = np.random.default_rng(20260806)
    model = GPT(vocab_size=11, d_model=24, n_heads=3, d_ff=48,
                n_blocks=2, max_T=9, seed=5)
    base = rng.integers(0, 11, size=(1, 9))
    reference, _ = model.forward(base)
    worst = 0.0
    for t in range(9):
        for _ in range(20):
            ids = base.copy()
            ids[0, t + 1:] = rng.integers(0, 11, size=8 - t)
            logits, _ = model.forward(ids)
            worst = max(worst, float(
                np.abs(logits[0, :t + 1] - reference[0, :t + 1]).max()))
    assert worst == 0.0, worst


# --------------------------------------------------------------------------
# gradients and the optimiser
# --------------------------------------------------------------------------

def test_batch_gradient_is_the_mean_of_its_rows():
    """grad(rows 0..B-1) == mean_b grad(row b), exactly.

    The loss averages over B·T predictions, and the backward pass is
    linear in d_logits, so a batch gradient must decompose. The property
    is the one that breaks first when a reshape mixes the batch and time
    axes the wrong way round.
    """
    rng = np.random.default_rng(20260807)
    for _ in range(6):
        V, T, B = 9, 5, 3
        ids = rng.integers(0, V, size=(B, T))
        tgt = rng.integers(0, V, size=(B, T))
        model = GPT(vocab_size=V, d_model=16, n_heads=2, d_ff=32,
                    n_blocks=2, max_T=T, seed=int(rng.integers(0, 10_000)))
        for p in model.params():
            p.zero_grad()
        model.loss_and_grads(ids, tgt)
        batched = [np.array(p.grad, copy=True) for p in model.params()]

        summed = [np.zeros_like(g) for g in batched]
        for b in range(B):
            for p in model.params():
                p.zero_grad()
            model.loss_and_grads(ids[b:b + 1], tgt[b:b + 1])
            for acc, p in zip(summed, model.params(), strict=True):
                acc += np.asarray(p.grad) / B
        for got, want in zip(batched, summed, strict=True):
            denom = max(1.0, float(np.abs(want).max()))
            assert float(np.abs(got - want).max()) / denom < 1e-12


def test_gradients_accumulate_instead_of_overwriting():
    """Two backward passes without zero_grad give exactly twice the gradient.

    Every `.grad` in the repo is written as `grad = (grad or 0) + delta`,
    including the embedding scatter-add. Accumulation is what lets a caller
    sum gradients over micro-batches, and an assignment where an addition
    belongs would silently keep only the last one.
    """
    rng = np.random.default_rng(20260808)
    V, T = 9, 6
    ids = rng.integers(0, V, size=(2, T))
    tgt = rng.integers(0, V, size=(2, T))
    model = GPT(vocab_size=V, d_model=16, n_heads=2, d_ff=32,
                n_blocks=2, max_T=T, seed=1)
    for p in model.params():
        p.zero_grad()
    model.loss_and_grads(ids, tgt)
    once = [np.array(p.grad, copy=True) for p in model.params()]
    model.loss_and_grads(ids, tgt)
    for single, p in zip(once, model.params(), strict=True):
        denom = max(1.0, float(np.abs(single).max()))
        assert float(np.abs(np.asarray(p.grad) - 2 * single).max()) / denom < 1e-13


@pytest.mark.parametrize("lr", [1e-4, 1e-3, 1e-2])
def test_adam_first_step_is_minus_lr_times_the_sign_of_the_gradient(lr):
    """With bias correction, step one is -lr·sign(g) regardless of |g|.

    Algorithm 1 of Kingma & Ba (2015) gives m̂₁ = g and v̂₁ = g² after the
    1/(1-β^t) corrections, so the update is -lr·g/(|g| + ε). The magnitude
    of the gradient cancels — which is exactly the claim bias correction
    exists to make true at t = 1, and the thing that silently breaks if the
    correction is dropped.
    """
    rng = np.random.default_rng(20260809)

    class Holder:
        def __init__(self, data):
            self.data = data
            self.grad = None

    for _ in range(50):
        n = int(rng.integers(1, 12))
        p = Holder(rng.standard_normal(n))
        before = p.data.copy()
        g = rng.standard_normal(n) * 10.0 ** rng.uniform(-4, 4)
        p.grad = g
        opt = AdamLite([p], lr=lr, eps=1e-12)
        opt.step()
        expected = before - lr * np.sign(g)
        assert float(np.abs(p.data - expected).max()) < 1e-9 * lr / 1e-4


def test_cached_decoding_equals_full_recompute_for_random_architectures():
    """Cached logits are the numbers a full forward pass would recompute.

    The causal mask makes K and V at position s independent of everything
    after s, so caching them is an identity rather than an approximation.
    Randomising depth, width and head count keeps that from being a fact
    about one lucky shape.
    """
    rng = np.random.default_rng(20260810)
    worst = 0.0
    for _ in range(12):
        heads = int(rng.integers(1, 5))
        d_head = int(rng.integers(2, 7))
        model = GPT(vocab_size=int(rng.integers(4, 20)),
                    d_model=heads * d_head, n_heads=heads,
                    d_ff=int(rng.integers(4, 33)),
                    n_blocks=int(rng.integers(1, 4)),
                    max_T=int(rng.integers(6, 14)),
                    seed=int(rng.integers(0, 10_000)))
        prompt = rng.integers(0, model.vocab_size, size=2)
        slow = model.generate(prompt, max_new=12, temperature=0.0,
                              use_cache=False)
        fast = model.generate(prompt, max_new=12, temperature=0.0,
                              use_cache=True)
        np.testing.assert_array_equal(slow, fast)
        # and the logits themselves, not just the argmax that survives them
        ids = [int(v) for v in prompt]
        _, cache = model.forward(np.array(ids)[None, :])
        kv = [TransformerBlock.kv_from_cache(c) for c in cache[1]]
        for pos in range(len(ids), model.max_T):
            ids.append(int(rng.integers(0, model.vocab_size)))
            step, kv = model.forward_step(np.array([[ids[-1]]]), pos, kv)
            full, _ = model.forward(np.array(ids)[None, :])
            worst = max(worst, float(np.abs(step[0] - full[0, -1]).max()))
    assert worst < 1e-12, worst
