"""Cached decoding must be the *same arithmetic*, not merely a close answer.

Incremental decoding is only a legitimate optimisation if feeding one token
against a cache produces the logits the full forward pass would have. These
tests pin that to float64 noise, including across the point where the
context window slides and the cache has to be rebuilt.
"""

import numpy as np
import pytest

from tfs.layers import TransformerBlock
from tfs.model import GPT


def model(max_T: int = 24) -> GPT:
    return GPT(vocab_size=13, d_model=32, n_heads=4, d_ff=64,
               n_blocks=3, max_T=max_T, seed=3)


def test_cached_logits_match_full_recompute_at_every_step():
    m = model()
    rng = np.random.default_rng(0)
    ids = [int(v) for v in rng.integers(0, 13, size=4)]
    _, cache = m.forward(np.array(ids)[None, :])
    kv = [TransformerBlock.kv_from_cache(c) for c in cache[1]]
    pos = len(ids)

    worst = 0.0
    for _ in range(20):
        nxt = int(rng.integers(0, 13))
        ids.append(nxt)
        cached, kv = m.forward_step(np.array([[nxt]]), pos, kv)
        pos += 1
        full, _ = m.forward(np.array(ids)[None, :])
        worst = max(worst, float(np.abs(cached[0] - full[0, -1]).max()))
    assert worst < 1e-12, f"cached decoding drifted: {worst:.3e}"


def test_greedy_generation_identical_with_and_without_cache():
    m = model()
    prompt = np.array([1, 2, 3, 4])
    slow = m.generate(prompt, max_new=15, temperature=0.0, use_cache=False)
    fast = m.generate(prompt, max_new=15, temperature=0.0, use_cache=True)
    np.testing.assert_array_equal(slow, fast)


def test_sampled_generation_identical_with_and_without_cache():
    """Same seed, same draws — the cache must not perturb the logits at all."""
    m = model()
    prompt = np.array([1, 2, 3, 4])
    slow = m.generate(prompt, max_new=15, temperature=0.8, use_cache=False,
                      rng=np.random.default_rng(7))
    fast = m.generate(prompt, max_new=15, temperature=0.8, use_cache=True,
                      rng=np.random.default_rng(7))
    np.testing.assert_array_equal(slow, fast)


def test_cache_is_rebuilt_when_the_context_window_slides():
    """Past max_T the window slides, every position id shifts, and the cached
    keys/values become invalid — generation must still be exact."""
    m = model(max_T=8)
    prompt = np.array([1, 2, 3])
    slow = m.generate(prompt, max_new=20, temperature=0.0, use_cache=False)
    fast = m.generate(prompt, max_new=20, temperature=0.0, use_cache=True)
    assert len(fast) == 23 > m.max_T
    np.testing.assert_array_equal(slow, fast)


def test_forward_step_validates_position_and_ids():
    m = model(max_T=8)
    _, cache = m.forward(np.array([[1, 2, 3]]))
    kv = [TransformerBlock.kv_from_cache(c) for c in cache[1]]
    with pytest.raises(ValueError, match="position"):
        m.forward_step(np.array([[1]]), 8, kv)
    with pytest.raises(ValueError, match="token ids"):
        m.forward_step(np.array([[99]]), 3, kv)
