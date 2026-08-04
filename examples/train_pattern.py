"""Train a tiny transformer on a periodic pattern + generate samples.

Training windows are sampled at *random offsets* into the sequence.
That detail is load-bearing: training on one fixed window lets the
position embeddings memorise "position 3 -> token 4", which falls apart
the moment generation slides the context window to a phase the model
never saw. See docs/positional-generalization.md for the full study.

Run:  PYTHONPATH=. python3 examples/train_pattern.py
"""

import argparse

import numpy as np

from tfs.model import GPT
from tfs.train import AdamLite

PERIOD = (1, 2, 3, 4, 5)
WINDOW = 8


def sample_windows(pattern: np.ndarray, window: int, batch: int,
                   rng: np.random.Generator):
    """(ids, targets) pairs starting at random offsets into `pattern`."""
    starts = rng.integers(0, len(pattern) - window, size=batch)
    ids = np.stack([pattern[s:s + window] for s in starts])
    tgt = np.stack([pattern[s + 1:s + window + 1] for s in starts])
    return ids, tgt


def train(steps: int = 400, lr: float = 5e-3, batch: int = 8, seed: int = 0,
          fixed_offset: bool = False, log=None) -> GPT:
    """Train the demo model on the period-5 pattern; returns the model.

    fixed_offset=True reproduces the positional-overfitting failure mode
    (always training on pattern[:8]) studied in the docs.
    """
    pattern = np.tile(PERIOD, 8)  # length 40
    rng = np.random.default_rng(seed)
    model = GPT(vocab_size=6, d_model=24, n_heads=3, d_ff=48,
                n_blocks=2, max_T=WINDOW, seed=seed)
    opt = AdamLite(model.params(), lr=lr)

    for step in range(steps):
        if fixed_offset:
            ids, tgt = pattern[None, :WINDOW], pattern[None, 1:WINDOW + 1]
        else:
            ids, tgt = sample_windows(pattern, WINDOW, batch, rng)
        opt.zero_grad()
        loss = model.loss_and_grads(ids, tgt)
        opt.step()
        if log is not None and (step % max(1, steps // 10) == 0
                                or step == steps - 1):
            log(f"{step:>5} {loss:>10.4f}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    args = ap.parse_args()

    print(f"{'step':>5} {'loss':>10}")
    model = train(steps=args.steps, lr=args.lr, log=print)

    # Greedy decoding: deterministic, so the printed output is reproducible.
    prompt = np.array([1, 2, 3])
    out = model.generate(prompt, max_new=20, temperature=0.0)
    expected = [PERIOD[i % len(PERIOD)] for i in range(len(out))]
    print(f"\nprompt    ={prompt.tolist()}")
    print(f"generation={out.tolist()}")
    print(f"target    ={expected}")
    print("exact match" if out.tolist() == expected else "MISMATCH")


if __name__ == "__main__":
    main()
