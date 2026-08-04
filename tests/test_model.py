"""GPT integration tests."""

import numpy as np

from tfs.model import GPT


def test_gpt_forward_shape():
    model = GPT(vocab_size=20, d_model=16, n_heads=4, d_ff=32,
                n_blocks=2, max_T=10, seed=0)
    ids = np.array([[0, 1, 2, 3, 4]])
    logits, _ = model.forward(ids)
    assert logits.shape == (1, 5, 20)


def test_gpt_loss_decreases_on_learnable_task():
    """Train on a deterministic next-token task; loss must actually fall.

    An earlier version of this test trained on *random* targets and then
    asserted only that the loss values were not None — it could not fail.
    This one trains on windows of a fixed cyclic sequence and requires a
    large real drop from the ~log(vocab) starting point.
    """
    from tfs.train import AdamLite
    vocab = 8
    rng = np.random.default_rng(0)
    base = np.tile(np.arange(vocab), 4)  # 0,1,...,7 repeating, length 32
    model = GPT(vocab_size=vocab, d_model=16, n_heads=2, d_ff=32,
                n_blocks=2, max_T=8, seed=0)
    opt = AdamLite(model.params(), lr=5e-3)

    losses = []
    for _ in range(150):
        starts = rng.integers(0, len(base) - 9, size=4)
        ids = np.stack([base[s:s + 8] for s in starts])
        tgt = np.stack([base[s + 1:s + 9] for s in starts])
        opt.zero_grad()
        losses.append(model.loss_and_grads(ids, tgt))
        opt.step()

    assert losses[0] > 1.5  # ~log(8) = 2.08 at init
    assert losses[-1] < 0.2  # the task is deterministic: must be learned
    assert losses[-1] < 0.1 * losses[0]


def test_gpt_learns_repeated_pattern():
    """If targets ARE the input shifted (next-token = next char of period-2 pattern)
    the model should drive loss well below random uniform."""
    from tfs.train import AdamLite
    vocab = 4
    # Period-2 sequence: 0,1,0,1,...
    base = np.tile([0, 1], 16)  # length 32
    model = GPT(vocab_size=vocab, d_model=16, n_heads=2, d_ff=32,
                n_blocks=1, max_T=8, seed=0)
    opt = AdamLite(model.params(), lr=1e-2)
    ids = base[None, :8]
    tgt = base[None, 1:9]

    losses = []
    for _ in range(200):
        for p in model.params():
            p.zero_grad()
        losses.append(model.loss_and_grads(ids, tgt))
        opt.step()

    # log(4) ~ 1.386. After training, loss should be well below half of that.
    assert losses[-1] < 0.5
