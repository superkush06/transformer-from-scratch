"""End-to-end gradient check on a small GPT instance."""

import numpy as np

from tfs.model import GPT
from tfs.ops import softmax_crossentropy


def main() -> None:
    np.random.default_rng(0)
    model = GPT(vocab_size=6, d_model=8, n_heads=2, d_ff=16,
                n_blocks=1, max_T=5, seed=0)
    ids = np.array([[0, 1, 2, 3]])
    tgt = np.array([[1, 2, 3, 4]])

    # Compute analytical grad on token_emb
    for p in model.params():
        p.zero_grad()
    model.loss_and_grads(ids, tgt)
    ad = model.token_emb.grad.copy()

    # Finite-diff grad on token_emb
    eps = 1e-5
    fd = np.zeros_like(ad)
    for i in range(model.token_emb.data.size):
        flat = model.token_emb.data.flatten()
        flat[i] += eps; model.token_emb.data = flat.reshape(model.token_emb.data.shape)
        logits, _ = model.forward(ids)
        lp, _ = softmax_crossentropy(logits, tgt)
        flat[i] -= 2 * eps; model.token_emb.data = flat.reshape(model.token_emb.data.shape)
        logits, _ = model.forward(ids)
        lm, _ = softmax_crossentropy(logits, tgt)
        flat[i] += eps; model.token_emb.data = flat.reshape(model.token_emb.data.shape)
        fd.flat[i] = (lp - lm) / (2 * eps)

    err = np.abs(ad - fd).max()
    rel = err / (np.abs(fd).max() + 1e-8)
    print(f"token_emb grad max abs error: {err:.6e}")
    print(f"token_emb grad max rel error: {rel:.6e}")
    print("PASS: rel < 1e-3" if rel < 1e-3 else "FAIL")


if __name__ == "__main__":
    main()
