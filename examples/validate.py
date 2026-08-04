"""Check this library against things that are not this library.

A test suite proves the code agrees with itself. This script proves it
agrees with something outside it: the definition of a derivative, a
reference implementation written from the paper rather than from
`tfs/`, a closed-form probability, an information-theoretic floor.

Every number `docs/validation.md` quotes is printed by this file. Run it:

    PYTHONPATH=. python3 examples/validate.py          # ~20 s

Rungs, in the order of the standard correctness ladder — gradients,
then a reference implementation, then an intervention on the masking,
then overfitting a tiny task on purpose:

  1  every parameter gradient vs central differences
  2  attention vs a naive O(n^2) implementation of the paper's equation
  3  causal masking vs an ablation of the future
  4  overfitting one batch to (essentially) zero loss
  5  the 1/sqrt(d_k) scale vs the variance it is supposed to fix
  6  the tanh GELU vs the exact Gaussian-CDF GELU        [disagrees]
  7  cross-entropy at uniform logits vs ln V
  8  LayerNorm output variance vs 'zero mean, unit variance'  [disagrees]
  9  Adam's first step vs -lr * sign(g)
 10  the parameter count vs the architecture's arithmetic
 11  held-out cross-entropy vs the source's entropy rate

Rows 6 and 8 are expected to disagree with the idealised claim, and they
are printed as disagreements rather than quietly rounded away. Both are
consequences of choices the library makes on purpose; the doc says which.

Every ok/DIFF verdict, those two included, is computed from the number
the check just measured — no row carries a constant verdict. Swap
`tfs.ops.gelu` for the exact `x*Phi(x)`, or drop the variance floor from
`tfs.ops.layernorm`, and the corresponding row flips to `ok` and the
count at the bottom moves with it. A summary line that could not do that
would not be worth printing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from examples.gradcheck import check_every_scalar
from tfs.attention import MultiHeadAttention
from tfs.model import GPT
from tfs.ops import layernorm, softmax_crossentropy
from tfs.train import AdamLite


@dataclass
class Check:
    """One row of the validation table."""
    claim: str
    ours: str
    reference: str
    source: str
    agrees: bool
    note: str = ""


# ---------------------------------------------------------------------------
# 1. every parameter gradient against the definition of a derivative
# ---------------------------------------------------------------------------

def check_gradients() -> Check:
    names, analytic, numeric = check_every_scalar()
    resolved = np.abs(numeric) > 1e-8
    rel = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-30)
    worst = float(rel[resolved].max())
    signs = int((np.sign(analytic) == np.sign(numeric)).sum())
    print(f"[1] gradients: {len(names):,} scalars over {len(set(names))} "
          f"tensors, {signs:,} signs agree, worst relative error {worst:.2e}")
    return Check(
        claim="max relative error, hand-derived gradient vs central difference",
        ours=f"{worst:.2e}",
        reference="0 (exact), with a float64 floor near eta^(2/3) ~ 4e-11",
        source="the definition of the derivative; Press et al., "
               "Numerical Recipes §5.7 for the achievable floor",
        agrees=worst < 1e-4,
        note="1,312 scalars, all 29 parameter tensors, eps=1e-5")


# ---------------------------------------------------------------------------
# 2. attention against a naive implementation of the published equation
# ---------------------------------------------------------------------------

def reference_attention(X, W_q, W_k, W_v, W_o, n_heads):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V, written out in loops.

    Deliberately naive: nested Python lists, `math.exp`, one dot product
    at a time, no max-subtraction, no broadcasting, no einsum. It shares
    no code path with `tfs.attention`, which is the point — if both are
    wrong they have to be wrong in the same way by coincidence.
    """
    B, T, D = len(X), len(X[0]), len(X[0][0])
    dh = D // n_heads
    ctx_all = [[[0.0] * D for _ in range(T)] for _ in range(B)]
    for b in range(B):
        for h in range(n_heads):
            lo = h * dh
            for i in range(T):
                q = [sum(X[b][i][m] * W_q[m][lo + c] for m in range(D))
                     for c in range(dh)]
                scores = []
                for j in range(i + 1):          # causal: j <= i only
                    k = [sum(X[b][j][m] * W_k[m][lo + c] for m in range(D))
                         for c in range(dh)]
                    scores.append(sum(q[c] * k[c] for c in range(dh))
                                  / math.sqrt(dh))
                total = sum(math.exp(s) for s in scores)
                weights = [math.exp(s) / total for s in scores]
                for j in range(i + 1):
                    v = [sum(X[b][j][m] * W_v[m][lo + c] for m in range(D))
                         for c in range(dh)]
                    for c in range(dh):
                        ctx_all[b][i][lo + c] += weights[j] * v[c]
    return np.array([[[sum(ctx_all[b][i][m] * W_o[m][d] for m in range(D))
                       for d in range(D)] for i in range(T)] for b in range(B)])


def check_attention_reference() -> Check:
    rng = np.random.default_rng(0)
    mha = MultiHeadAttention(d_model=8, n_heads=2, rng=rng)
    X = rng.standard_normal((2, 6, 8))
    ours, _ = mha.forward(X)
    ref = reference_attention(X.tolist(), mha.W_q.data.tolist(),
                              mha.W_k.data.tolist(), mha.W_v.data.tolist(),
                              mha.W_o.data.tolist(), 2)
    worst = float(np.abs(ours - ref).max())
    print(f"[2] attention: max |vectorised - naive| = {worst:.2e} "
          f"over a (2, 6, 8) input; outputs are O({np.abs(ours).max():.2f})")
    return Check(
        claim="max absolute difference, vectorised attention vs a loop "
              "implementation of the paper's equation",
        ours=f"{worst:.2e}",
        reference="0 (same function)",
        source="Vaswani et al. (2017), §3.2.1, scaled dot-product attention",
        agrees=worst < 1e-12,
        note="2 heads, T=6; the reference shares no code with tfs/")


# ---------------------------------------------------------------------------
# 3. causal masking, by intervention
# ---------------------------------------------------------------------------

def check_causal_mask(trials_per_position: int = 40) -> Check:
    """Rewrite the future; the past must not move by a single bit."""
    rng = np.random.default_rng(4)
    V, T = 11, 9
    model = GPT(vocab_size=V, d_model=24, n_heads=3, d_ff=48,
                n_blocks=2, max_T=T, seed=5)
    base = rng.integers(0, V, size=(1, T))
    reference, _ = model.forward(base)
    worst, trials = 0.0, 0
    for t in range(T):
        for k in range(trials_per_position):
            ids = base.copy()
            future = ids[0, t + 1:]
            if k % 2 == 0:                       # resample the suffix
                ids[0, t + 1:] = rng.integers(0, V, size=future.size)
            else:                                # or just permute it
                shuffled = future.copy()
                rng.shuffle(shuffled)
                ids[0, t + 1:] = shuffled
            logits, _ = model.forward(ids)
            worst = max(worst, float(np.abs(
                logits[0, :t + 1] - reference[0, :t + 1]).max()))
            trials += 1
    print(f"[3] causal mask: {trials} interventions on the future, "
          f"largest change to any earlier logit = {worst!r}")
    return Check(
        claim="largest change to a logit at position <= t when every token "
              "after t is rewritten",
        ours=f"{worst:.1f}",
        reference="0 exactly",
        source="the causal-mask definition; -1e9 underflows to exactly zero "
               "weight in float64",
        agrees=worst == 0.0,
        note=f"{trials} interventions (resample and permute), T=9")


# ---------------------------------------------------------------------------
# 4. overfit one batch
# ---------------------------------------------------------------------------

def check_overfit_one_batch(steps: int = 4000) -> Check:
    """Karpathy's second rung: if it cannot memorise one batch, stop.

    A single sequence of random targets, so every prefix is distinct and
    the task is consistent — the model only has to memorise 16 arbitrary
    context -> token mappings. If a manual backward pass has a subtle sign
    error somewhere, this is where it shows up as a floor the loss will
    not go below.
    """
    rng = np.random.default_rng(7)
    V, T = 16, 16
    ids = rng.integers(0, V, size=(1, T))
    tgt = rng.integers(0, V, size=(1, T))
    model = GPT(vocab_size=V, d_model=32, n_heads=4, d_ff=64,
                n_blocks=2, max_T=T, seed=0)
    opt = AdamLite(model.params(), lr=1e-2)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = model.loss_and_grads(ids, tgt)
        first = loss if first is None else first
        opt.step()
    logits, _ = model.forward(ids)
    acc = float((logits.argmax(-1) == tgt).mean())
    print(f"[4] overfit: loss {first:.4f} -> {loss:.2e} in {steps} steps "
          f"(ln {V} = {math.log(V):.4f}); argmax accuracy {acc:.3f}")
    return Check(
        claim="final loss and argmax accuracy after memorising one batch",
        ours=f"{loss:.2e} / {acc:.3f}",
        reference=f"0 / 1.000 (the task is deterministic; start is ln {V} "
                  f"= {math.log(V):.3f})",
        source="Karpathy (2019), A Recipe for Training Neural Networks — "
               "'overfit a single batch'",
        agrees=loss < 1e-5 and acc == 1.0,
        note=f"1x{T} sequence, random targets, {steps} Adam steps at lr 1e-2")


# ---------------------------------------------------------------------------
# 5. why the scores are divided by sqrt(d_k)
# ---------------------------------------------------------------------------

def check_score_scaling(d_k: int = 64, n: int = 200_000) -> Check:
    """Var(q·k) = d_k for iid unit-variance components; the scale undoes it."""
    rng = np.random.default_rng(2)
    q = rng.standard_normal((n, d_k))
    k = rng.standard_normal((n, d_k))
    raw = float((q * k).sum(axis=1).var())
    scaled = raw / d_k
    print(f"[5] score scale: Var(q·k) = {raw:.2f} for d_k = {d_k}; "
          f"after dividing by sqrt(d_k), Var = {scaled:.4f}")
    return Check(
        claim="variance of an unscaled dot product of two unit-variance "
              f"vectors in {d_k} dimensions",
        ours=f"{raw:.2f}",
        reference=f"d_k = {d_k}",
        source="Vaswani et al. (2017), §3.2.1 — the argument for the "
               "1/sqrt(d_k) factor",
        agrees=abs(raw - d_k) < 4 * d_k / math.sqrt(n),
        note=f"{n:,} Monte-Carlo draws; scaled variance {scaled:.4f}")


# ---------------------------------------------------------------------------
# 6. the tanh GELU against the exact one  [expected to disagree]
# ---------------------------------------------------------------------------

def check_gelu_against_exact() -> Check:
    """tfs.ops.gelu is the tanh approximation, not x*Phi(x). Measure the gap."""
    from tfs.ops import gelu
    x = np.linspace(-6.0, 6.0, 200_001)
    exact = x * 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    gap = np.abs(gelu(x) - exact)
    worst = float(gap.max())
    where = float(x[int(np.argmax(gap))])
    verdict = ("this is a real disagreement" if worst > 1e-12
               else "the exact form is implemented now")
    print(f"[6] GELU: max |tanh form - x*Phi(x)| = {worst:.2e} "
          f"at x = {where:+.3f}  ({verdict})")
    return Check(
        claim="max absolute difference between tfs.ops.gelu and the exact "
              "Gaussian-CDF GELU on [-6, 6]",
        ours=f"{worst:.2e} at x = {where:+.2f}",
        reference="0 if the exact form were implemented",
        source="Hendrycks & Gimpel (2016), Gaussian Error Linear Units — "
               "both the exact x*Phi(x) and the tanh approximation",
        # Derived, not hardcoded: swap tfs.ops.gelu for the exact form and
        # this row flips to `ok` on its own.
        agrees=worst < 1e-12,
        note="deliberate: the repo ships the tanh approximation, as GPT-2 "
             "does, and its derivative is the one written out in tfs/ops.py")


# ---------------------------------------------------------------------------
# 7. cross-entropy at uniform logits
# ---------------------------------------------------------------------------

def check_uniform_cross_entropy(V: int = 17) -> Check:
    loss, _ = softmax_crossentropy(np.zeros((5, V)), np.arange(5) % V)
    print(f"[7] uniform cross-entropy: {loss!r} vs ln {V} = {math.log(V)!r}")
    return Check(
        claim=f"cross-entropy of a uniform distribution over {V} classes",
        ours=f"{loss:.15f}",
        reference=f"ln {V} = {math.log(V):.15f}",
        source="closed form: -ln(1/V)",
        agrees=loss == math.log(V),
        note="exact to the last bit of float64")


# ---------------------------------------------------------------------------
# 8. LayerNorm's output variance  [disagrees with the idealised claim]
# ---------------------------------------------------------------------------

def check_layernorm_variance(eps: float = 1e-5) -> Check:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((256, 32)) * 2.0
    out, _ = layernorm(x, np.ones(32), np.zeros(32), eps=eps)
    measured = float(out.var(axis=-1).mean())
    var = x.var(axis=-1)
    predicted = float(np.mean(var / (var + eps)))
    print(f"[8] LayerNorm: mean output variance {measured:.10f}; "
          f"var/(var+eps) predicts {predicted:.10f}; the textbook claim is 1")
    return Check(
        claim="mean output variance of LayerNorm with gamma=1, beta=0",
        ours=f"{measured:.10f}",
        reference="1.0 exactly — the usual 'zero mean, unit variance' claim",
        source="Ba, Kiros & Hinton (2016), Layer Normalization",
        # Derived, not hardcoded: drop the variance floor and this row
        # flips to `ok` on its own.
        agrees=abs(measured - 1.0) < 1e-12,
        note=f"matches var/(var+eps) = {predicted:.10f} to "
             f"{abs(measured - predicted):.0e}, so the gap is the variance "
             f"floor and nothing else")


# ---------------------------------------------------------------------------
# 9. Adam's first step
# ---------------------------------------------------------------------------

def check_adam_first_step(lr: float = 1e-3) -> Check:
    """Bias correction makes step one exactly -lr * sign(g), for any |g|."""
    rng = np.random.default_rng(3)

    class Holder:
        def __init__(self, data):
            self.data, self.grad = data, None

    adam_eps = 1e-14
    worst_sign, worst_exact = 0.0, 0.0
    for _ in range(200):
        p = Holder(rng.standard_normal(8))
        before = p.data.copy()
        g = rng.standard_normal(8) * 10.0 ** rng.uniform(-6, 6)
        p.grad = g
        AdamLite([p], lr=lr, eps=adam_eps).step()
        step = p.data - before
        worst_sign = max(worst_sign,
                         float(np.abs(step + lr * np.sign(g)).max()))
        # the exact prediction keeps Adam's epsilon instead of dropping it
        exact = -lr * g / (np.abs(g) + adam_eps)
        worst_exact = max(worst_exact, float(np.abs(step - exact).max()))
    print(f"[9] Adam: max |step_1 - (-lr*g/(|g|+eps))| = {worst_exact:.2e}; "
          f"against the eps -> 0 form -lr*sign(g) it is {worst_sign:.2e}")
    return Check(
        claim="first Adam step, for gradients spanning 12 orders of magnitude",
        ours=f"|error| <= {worst_exact:.2e}",
        reference=f"-lr*g/(|g| + eps), i.e. -lr*sign(g) = {lr:.0e} as eps -> 0",
        source="Kingma & Ba (2015), Adam, Algorithm 1 — after bias "
               "correction m1_hat = g and v1_hat = g^2",
        agrees=worst_exact < 1e-15,
        note=f"the magnitude of the gradient cancels; the {worst_sign:.0e} "
             f"residual against plain sign(g) is exactly lr*eps/(|g|+eps)")


# ---------------------------------------------------------------------------
# 10. the parameter count
# ---------------------------------------------------------------------------

def check_parameter_count() -> Check:
    V, D, H, F, L, T = 6, 24, 3, 48, 2, 8
    model = GPT(vocab_size=V, d_model=D, n_heads=H, d_ff=F,
                n_blocks=L, max_T=T, seed=0)
    counted = sum(p.data.size for p in model.params())
    per_block = 4 * D * D + 2 * D * F + F + D + 4 * D
    closed = V * D + T * D + 2 * D + L * per_block + D * V
    print(f"[10] parameters: {counted:,} counted, {closed:,} from the "
          f"architecture ({L} x {per_block:,} in blocks)")
    return Check(
        claim="parameter count of the README's demo model",
        ours=f"{counted:,}",
        reference=f"{closed:,}",
        source="closed form: VD + T_max·D + 2D + L(4D² + 2D·d_ff + d_ff + 5D) "
               "+ DV",
        agrees=counted == closed,
        note=f"V={V}, D={D}, heads={H}, d_ff={F}, blocks={L}, max_T={T}")


# ---------------------------------------------------------------------------
# 11. held-out cross-entropy against an information-theoretic floor
# ---------------------------------------------------------------------------

def check_entropy_rate() -> Check:
    """The strongest check available: a floor the model must not go under."""
    from examples.regime_handoff import (
        bigram_cross_entropy,
        entropy_rates,
        fit,
        model_cross_entropy,
        oracle_cross_entropy,
        simulate,
        transition_tensor,
    )
    P = transition_tensor()
    h2, h1 = entropy_rates(P)
    train_seq = simulate(P, 20_000, seed=1)
    test_seq = simulate(P, 4_000, seed=2)
    model = fit(train_seq)
    ours = model_cross_entropy(model, test_seq)
    oracle = oracle_cross_entropy(P, test_seq)
    bigram = bigram_cross_entropy(train_seq, test_seq)
    print(f"[11] entropy rate: model {ours:.4f}, oracle {oracle:.4f}, "
          f"bigram {bigram:.4f} nats/label; closed-form rate {h2:.4f}, "
          f"H(X|X-1) {h1:.4f}")
    return Check(
        claim="held-out cross-entropy on a second-order Markov source, "
              "against the Bayes floor",
        ours=f"{ours:.4f} nats/label",
        reference=f"{oracle:.4f} (the source scoring the same tokens); "
                  f"H = {h2:.4f} in the limit",
        source="Cover & Thomas, Elements of Information Theory (2nd ed.), "
               "§4.2 — the entropy rate of a stationary Markov chain",
        agrees=ours >= oracle and ours - oracle < 0.02,
        note=f"the order-1 baseline scores {bigram:.4f}; its handicap "
             f"I(X;X-2|X-1) = {h1 - h2:.4f} nats, and it loses "
             f"{bigram - ours:.4f}")


CHECKS = (
    check_gradients,
    check_attention_reference,
    check_causal_mask,
    check_overfit_one_batch,
    check_score_scaling,
    check_gelu_against_exact,
    check_uniform_cross_entropy,
    check_layernorm_variance,
    check_adam_first_step,
    check_parameter_count,
    check_entropy_rate,
)


def run_all() -> list[Check]:
    return [fn() for fn in CHECKS]


def main() -> None:
    print("validating tfs against references outside it\n")
    results = run_all()

    def wrap(text: str, indent: int, width: int = 74):
        words, line, lines = text.split(), "", []
        for w in words:
            if line and len(line) + 1 + len(w) > width:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
        pad = " " * indent
        return f"\n{pad}".join(lines)

    print("\n" + "=" * 90)
    for i, c in enumerate(results, 1):
        mark = "ok" if c.agrees else "DIFF"
        print(f"[{i:>2}] {mark:<4} {wrap(c.claim, 10)}")
        print(f"{'':10}ours       {wrap(c.ours, 21)}")
        print(f"{'':10}reference  {wrap(c.reference, 21)}")
        print(f"{'':10}source     {wrap(c.source, 21)}")
        if c.note:
            print(f"{'':10}note       {wrap(c.note, 21)}")
        print("-" * 90)
    disagree = [str(i) for i, c in enumerate(results, 1) if not c.agrees]
    agree = len(results) - len(disagree)
    print(f"{agree}/{len(results)} rungs agree with their reference. "
          f"Row(s) {', '.join(disagree)} disagree on purpose; "
          f"docs/validation.md says why.")


if __name__ == "__main__":
    main()
