"""Check every hand-derived gradient in the model against the definition.

This is the whole claim of the repository, made executable: for a 2-block
GPT we perturb *every scalar of every parameter tensor* by +/- eps, measure
how the loss actually moved, and compare that with the gradient the manual
backward pass computed.

    PYTHONPATH=. python3 examples/gradcheck.py

The model is deliberately tiny (1,312 parameters) because a central
difference costs two full forward passes per coordinate — the point here is
completeness, not scale. `tests/test_gradcheck.py` runs the same check on
sampled coordinates so CI stays fast; docs/figures.py plots it.

Token ids repeat within a row on purpose: the embedding gradient is a
scatter-add, and repeats are where a naive implementation overwrites
instead of accumulating.
"""

import numpy as np

from tfs.model import GPT
from tfs.ops import softmax_crossentropy

IDS = np.array([[1, 2, 1, 2, 5], [3, 3, 0, 1, 1]])
TGT = np.array([[2, 1, 2, 5, 6], [3, 0, 1, 1, 4]])
TOLERANCE = 1e-4


def build_model() -> GPT:
    return GPT(vocab_size=7, d_model=8, n_heads=2, d_ff=16,
               n_blocks=2, max_T=6, seed=0)


def loss_fn(model: GPT):
    """A zero-argument loss, so the finite-difference loop stays readable."""
    def loss() -> float:
        logits, _ = model.forward(IDS)
        value, _ = softmax_crossentropy(logits, TGT)
        return value
    return loss


def check_every_scalar(eps: float = 1e-5):
    """Return (names, analytic, central_difference) over all coordinates.

    One backward pass fills every `.grad`; after that the loop only reads
    gradients and perturbs `.data`, restoring it each time.
    """
    model = build_model()
    for p in model.params():
        p.zero_grad()
    model.loss_and_grads(IDS, TGT)
    loss = loss_fn(model)

    names, analytic, numeric = [], [], []
    for name, param in model.named_params():
        for c in range(param.data.size):
            original = param.data.flat[c]
            param.data.flat[c] = original + eps
            up = loss()
            param.data.flat[c] = original - eps
            down = loss()
            param.data.flat[c] = original
            names.append(name)
            numeric.append((up - down) / (2 * eps))
            analytic.append(float(param.grad.flat[c]))
    return np.array(names), np.array(analytic), np.array(numeric)


def main() -> None:
    names, analytic, numeric = check_every_scalar()
    # Below ~1e-8 the central difference is mostly cancellation noise, so a
    # relative error there says more about float64 than about the gradient.
    resolved = np.abs(numeric) > 1e-8
    rel = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-30)

    print(f"2-block GPT: {len(set(names))} parameter tensors, "
          f"{len(names):,} scalars, central differences at eps=1e-5\n")
    print(f"  {'tensor':<20}{'n':>6}{'max abs err':>14}{'max rel err':>14}")
    print(f"  {'-' * 54}")
    for name in dict.fromkeys(names):
        sel = names == name
        print(f"  {name:<20}{sel.sum():>6}"
              f"{np.abs(analytic[sel] - numeric[sel]).max():>14.2e}"
              f"{rel[sel & resolved].max():>14.2e}")
    print(f"  {'-' * 54}")
    worst = rel[resolved].max()
    print(f"  {'every tensor':<20}{len(names):>6}"
          f"{np.abs(analytic - numeric).max():>14.2e}{worst:>14.2e}")
    signs = int((np.sign(analytic) == np.sign(numeric)).sum())
    print(f"\nsigns agree on {signs:,}/{len(names):,} coordinates")
    print("PASS" if worst < TOLERANCE else "FAIL",
          f"(tolerance {TOLERANCE:.0e})")


if __name__ == "__main__":
    main()
