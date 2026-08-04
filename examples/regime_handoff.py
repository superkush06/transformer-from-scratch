"""A worked hand-off: regime labels in, a predictive distribution out.

This repository is the deep-learning-internals layer of a larger set of
NumPy libraries. It is rarely the whole pipeline. Upstream, something
turns a continuous series into discrete labels — a Gaussian HMM decoded
with Viterbi, say, of the kind `regimes` implements. Downstream,
something consumes a distribution over the next label — a sizing rule
(`kelly-bet`) or a tail-risk model (`risk`) wants probabilities, not a
point forecast. In the middle sits a sequence model whose job is
`P(next label | history)`.

Nothing here imports the sibling libraries. The upstream stage is
inlined as a second-order Markov source over four labels, because a
synthetic source is the only kind whose answer we can *check*: its
conditional distribution, its entropy rate, and the gap between an
order-2 and an order-1 predictor are all available in closed form. So
this doubles as the hardest validation in the repo — the model is
scored against an information-theoretic floor it must not beat.

    PYTHONPATH=. python3 examples/regime_handoff.py     # ~15 s

Labels: 0 calm-up, 1 calm-down, 2 stress-up, 3 stress-down. The source
persists the volatility regime and continues the move direction, with
both effects depending on the two previous bars — the minimum structure
that a bigram cannot capture and a transformer can.
"""

from __future__ import annotations

import math

import numpy as np

from tfs.model import GPT
from tfs.ops import softmax
from tfs.train import AdamLite

LABELS = ("calm-up", "calm-dn", "stress-up", "stress-dn")
WINDOW = 12


def transition_tensor() -> np.ndarray:
    """P[a, b, c] = P(next = c | previous two labels were a then b).

    Two independent coin flips per step, each conditioned on history:
    the regime (calm/stress) persists, and persists harder when the bar
    before last agreed; the direction (up/down) continues, more strongly
    inside the stressed regime.
    """
    P = np.zeros((4, 4, 4))
    for a in range(4):
        for b in range(4):
            calm_a, calm_b = a < 2, b < 2
            up_b = b % 2 == 0
            p_calm = 0.93 if calm_b else 0.10
            if calm_a != calm_b:  # the regime only just switched
                p_calm = 0.65 if calm_b else 0.35
            p_up = (0.62 if up_b else 0.38) if calm_b else (0.78 if up_b else 0.22)
            for c in range(4):
                calm_c, up_c = c < 2, c % 2 == 0
                P[a, b, c] = ((p_calm if calm_c else 1 - p_calm)
                              * (p_up if up_c else 1 - p_up))
    assert np.allclose(P.sum(-1), 1.0)
    return P


def stationary_pair_distribution(P: np.ndarray) -> np.ndarray:
    """pi[a, b], the stationary law of consecutive label pairs.

    An order-2 chain on 4 labels is an order-1 chain on the 16 ordered
    pairs, so the stationary distribution is the leading left eigenvector
    of that pair-to-pair matrix.
    """
    M = np.zeros((16, 16))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                M[a * 4 + b, b * 4 + c] = P[a, b, c]
    values, vectors = np.linalg.eig(M.T)
    lead = int(np.argmax(values.real))
    pi = np.abs(vectors[:, lead].real)
    return (pi / pi.sum()).reshape(4, 4)


def entropy_rates(P: np.ndarray) -> tuple[float, float]:
    """(order-2 entropy rate, order-1 entropy) in nats.

    For a stationary Markov chain the entropy rate is the conditional
    entropy of one step given the state, so here it is H(X_t | X_{t-2},
    X_{t-1}) — the cross-entropy no predictor can beat. The order-1 value
    H(X_t | X_{t-1}) is what a bigram model is stuck with; the difference
    is the conditional mutual information I(X_t ; X_{t-2} | X_{t-1}), and
    it is exactly the edge available to anything that looks back two bars.
    """
    pi = stationary_pair_distribution(P)
    h2 = -sum(pi[a, b] * sum(P[a, b, c] * math.log(P[a, b, c])
                             for c in range(4))
              for a in range(4) for b in range(4))
    pi_b = pi.sum(axis=0)
    h1 = 0.0
    for b in range(4):
        row = sum(pi[a, b] * P[a, b] for a in range(4)) / pi_b[b]
        h1 -= pi_b[b] * sum(row[c] * math.log(row[c]) for c in range(4))
    return float(h2), float(h1)


def simulate(P: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Draw a label sequence from the source. Burn-in is the first 2 steps."""
    rng = np.random.default_rng(seed)
    s = [0, 0]
    for _ in range(n + 200):
        s.append(int(rng.choice(4, p=P[s[-2], s[-1]])))
    return np.array(s[202:])


def fit(train_seq: np.ndarray, steps: int = 2500, lr: float = 3e-3,
        batch: int = 32, seed: int = 0, log=None) -> GPT:
    """Train the sequence model on random windows of the label stream."""
    model = GPT(vocab_size=4, d_model=32, n_heads=4, d_ff=64,
                n_blocks=2, max_T=WINDOW, seed=seed)
    opt = AdamLite(model.params(), lr=lr)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        starts = rng.integers(0, len(train_seq) - WINDOW - 1, size=batch)
        ids = np.stack([train_seq[s:s + WINDOW] for s in starts])
        tgt = np.stack([train_seq[s + 1:s + WINDOW + 1] for s in starts])
        opt.zero_grad()
        loss = model.loss_and_grads(ids, tgt)
        opt.step()
        if log is not None and (step % max(1, steps // 6) == 0
                                or step == steps - 1):
            log(f"{step:>5} {loss:>9.4f}")
    return model


def oracle_cross_entropy(P: np.ndarray, seq: np.ndarray) -> float:
    """The Bayes floor on *these tokens*: the source scoring its own draw.

    Comparing against this rather than against the entropy rate removes
    sampling noise — the model and the oracle are graded on the same
    realisation, so any gap is the model's, not the draw's.
    """
    return float(np.mean([-math.log(P[seq[t - 2], seq[t - 1], seq[t]])
                          for t in range(2, len(seq))]))


def bigram_cross_entropy(train_seq: np.ndarray, test_seq: np.ndarray) -> float:
    """Maximum-likelihood order-1 baseline, fitted on train, scored on test."""
    counts = np.zeros((4, 4))
    for t in range(1, len(train_seq)):
        counts[train_seq[t - 1], train_seq[t]] += 1.0
    table = counts / counts.sum(axis=1, keepdims=True)
    return float(np.mean([-math.log(table[test_seq[t - 1], test_seq[t]])
                          for t in range(2, len(test_seq))]))


def model_cross_entropy(model: GPT, test_seq: np.ndarray) -> float:
    """Held-out cross-entropy, counting only positions with 2+ tokens of context.

    The first two predictions in each window are excluded because the
    source itself is undefined there: with fewer than two previous labels
    the oracle has no conditional to offer, so scoring them would compare
    the model against nothing.
    """
    nll = []
    stride = WINDOW - 2
    for s in range(0, len(test_seq) - WINDOW - 1, stride):
        ids = test_seq[s:s + WINDOW][None, :]
        tgt = test_seq[s + 1:s + WINDOW + 1]
        logits, _ = model.forward(ids)
        probs = softmax(logits[0], axis=-1)
        nll.extend(-np.log(probs[np.arange(WINDOW), tgt])[2:])
    return float(np.mean(nll))


def predictive_table(model: GPT, P: np.ndarray):
    """Model vs source conditional for all 16 two-label contexts.

    Returns rows of (a, b, model row, source row, total variation,
    stationary weight). Total variation is the right yardstick for a
    downstream consumer: it bounds how wrong any bounded decision rule
    built on the distribution can be. The stationary weight is carried
    alongside because the two are related — the contexts the source
    almost never visits are the ones the model has least reason to get
    right, and reporting the worst TV without its weight overstates the
    error a caller would actually meet.
    """
    pi = stationary_pair_distribution(P)
    rows = []
    for a in range(4):
        for b in range(4):
            logits, _ = model.forward(np.array([[a, b]]))
            p = softmax(logits[0, -1])
            d = 0.5 * float(np.abs(p - P[a, b]).sum())
            rows.append((a, b, p, P[a, b], d, float(pi[a, b])))
    tv = np.array([r[4] for r in rows])
    weights = np.array([r[5] for r in rows])
    return rows, tv, weights


def main() -> None:
    P = transition_tensor()
    h2, h1 = entropy_rates(P)
    train_seq = simulate(P, 20_000, seed=1)
    test_seq = simulate(P, 4_000, seed=2)

    mix = np.bincount(train_seq, minlength=4) / len(train_seq)
    print("upstream: 24,000 regime labels from a second-order source")
    print("  label mix (train): "
          + ", ".join(f"{n} {v:.3f}"
                        for n, v in zip(LABELS, mix, strict=True)))
    print(f"\n{'step':>5} {'loss':>9}")
    model = fit(train_seq, log=print)

    oracle = oracle_cross_entropy(P, test_seq)
    bigram = bigram_cross_entropy(train_seq, test_seq)
    ours = model_cross_entropy(model, test_seq)

    print("\nheld-out cross-entropy, nats/label (lower is better)")
    print(f"  uniform over 4 labels                {math.log(4):>8.4f}")
    print(f"  order-1 entropy H(X|X-1)   [closed]  {h1:>8.4f}")
    print(f"  bigram MLE, fitted + scored          {bigram:>8.4f}")
    print(f"  this model                           {ours:>8.4f}")
    print(f"  oracle: the source itself            {oracle:>8.4f}")
    print(f"  entropy rate H(X|X-1,X-2)  [closed]  {h2:>8.4f}")
    print(f"\n  excess over the oracle:  model {ours - oracle:+.4f}"
          f"   bigram {bigram - oracle:+.4f}")
    print(f"  the bigram's handicap is I(X; X-2 | X-1) = {h1 - h2:.4f} "
          f"nats [closed form]")

    rows, tv, weights = predictive_table(model, P)
    worst = rows[int(np.argmax(tv))]
    print("\ntotal variation to the true conditional, over all 16 contexts")
    print(f"  max {tv.max():.4f}   mean {tv.mean():.4f}   "
          f"stationary-weighted mean {float(tv @ weights):.4f}")
    print(f"  worst context: ({LABELS[worst[0]]}, {LABELS[worst[1]]}), "
          f"which carries {worst[5] * 100:.1f}% of the stationary mass")

    print(f"\n{'context':>22}  {'model P(next)':>29}  {'source P(next)':>29}"
          f"  {'weight':>7}")
    for a, b, p, truth, _, w in rows[:4]:
        ctx = f"{LABELS[a]}, {LABELS[b]}"
        fmt = "[" + " ".join(f"{v:.3f}" for v in p) + "]"
        ref = "[" + " ".join(f"{v:.3f}" for v in truth) + "]"
        print(f"{ctx:>22}  {fmt:>29}  {ref:>29}  {w:>7.3f}")

    print("\nDownstream takes one of those rows — a distribution over the "
          "next regime — and\nsizes a position with it. The number that "
          "matters there is the total variation,\nnot the training loss.")


if __name__ == "__main__":
    main()
