"""Train a tiny transformer on a periodic pattern + generate samples.

Run:  PYTHONPATH=. python3 examples/train_pattern.py
"""

import argparse

import numpy as np

from tfs.model import GPT
from tfs.train import AdamLite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    args = ap.parse_args()

    vocab = 6
    pattern = np.tile([1, 2, 3, 4, 5], 6)  # period-5 pattern, len 30
    rng = np.random.default_rng(0)
    model = GPT(vocab_size=vocab, d_model=24, n_heads=3, d_ff=48,
                n_blocks=2, max_T=8, seed=0)
    opt = AdamLite(model.params(), lr=args.lr)

    ids = pattern[None, :8]
    tgt = pattern[None, 1:9]

    print(f"{'step':>5} {'loss':>10}")
    for step in range(args.steps):
        for p in model.params():
            p.zero_grad()
        loss = model.loss_and_grads(ids, tgt)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"{step:>5} {loss:>10.4f}")

    # Sample from a small prompt and see if we get the pattern
    prompt = np.array([1, 2, 3])
    out = model.generate(prompt, max_new=10, temperature=0.5, rng=rng)
    print(f"\nprompt={prompt.tolist()}")
    print(f"generation={out.tolist()}")
    expected = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]
    print(f"target={expected}")


if __name__ == "__main__":
    main()
