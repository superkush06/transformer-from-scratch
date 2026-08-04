"""Fixed-offset vs random-offset training: a positional-generalization study.

Trains the demo model both ways, teacher-forces the true periodic
sequence through each, and records per-position greedy next-token
accuracy. The fixed-offset model is perfect inside its training window
and collapses at position 9 — the first prediction whose context window
has changed phase relative to anything it ever saw. The random-offset
model has seen every phase at every position, so its curve stays at 1.0.

Writes docs/positional_generalization.png (requires matplotlib) via the
shared plotting code in docs/figures.py, so this figure carries the same
fonts and palette as the other two.

Run:  PYTHONPATH=. python3 examples/positional_generalization.py
"""

import numpy as np

from examples.train_pattern import PERIOD, WINDOW, train

SEQ_LEN = 40
PROMPT_LEN = 3
GEN_LEN = 20


def per_position_accuracy(model, seq: np.ndarray) -> np.ndarray:
    """Greedy next-token correctness at each position, teacher-forced.

    At position t the model sees the *true* last-WINDOW tokens (exactly
    what sliding-window generation would feed it if all its previous
    predictions were right) and must predict seq[t].
    """
    hits = []
    for t in range(PROMPT_LEN, len(seq)):
        ctx = seq[max(0, t - WINDOW):t][None, :]
        logits, _ = model.forward(ctx)
        hits.append(float(np.argmax(logits[0, -1]) == seq[t]))
    return np.array(hits)


def study(plot: bool = True):
    """Train both models, print the accuracy table, draw the figure."""
    seq = np.tile(PERIOD, SEQ_LEN // len(PERIOD))
    print("training fixed-offset model (always the same window) ...")
    fixed = train(steps=400, fixed_offset=True, log=None)
    print("training random-offset model ...")
    rand = train(steps=400, log=None)

    positions = np.arange(PROMPT_LEN, SEQ_LEN)
    acc_fixed = per_position_accuracy(fixed, seq)
    acc_rand = per_position_accuracy(rand, seq)

    # Predicting position t uses window seq[t-8:t]; the first window the
    # fixed-offset model has never seen is t = 9 (seq[1:9]).
    inside = positions <= WINDOW
    print(f"\n{'positions':>10} {'fixed-offset':>13} {'random-offset':>14}")
    print(f"{'3-8':>10} {acc_fixed[inside].mean():>13.2f} "
          f"{acc_rand[inside].mean():>14.2f}")
    print(f"{'9-39':>10} {acc_fixed[~inside].mean():>13.2f} "
          f"{acc_rand[~inside].mean():>14.2f}")

    prompt = np.array(seq[:PROMPT_LEN])
    gen_fixed = fixed.generate(prompt, max_new=GEN_LEN, temperature=0.0)
    gen_rand = rand.generate(prompt, max_new=GEN_LEN, temperature=0.0)
    print(f"\nfixed-offset  greedy: {gen_fixed.tolist()}")
    print(f"random-offset greedy: {gen_rand.tolist()}")

    if plot:
        from docs.figures import fig_positional
        fig_positional(positions, acc_fixed, acc_rand, gen_fixed, gen_rand,
                       seq[:len(gen_rand)], WINDOW)
    return acc_fixed, acc_rand, gen_fixed, gen_rand


if __name__ == "__main__":
    study()
