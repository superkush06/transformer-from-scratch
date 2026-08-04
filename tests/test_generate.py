"""Sampling / input-validation edge cases for GPT.generate and forward."""

import numpy as np
import pytest

from tfs.model import GPT


def small_model() -> GPT:
    return GPT(vocab_size=6, d_model=16, n_heads=2, d_ff=32,
               n_blocks=1, max_T=8, seed=0)


def test_forward_rejects_out_of_vocab_ids():
    model = small_model()
    with pytest.raises(ValueError, match="token ids"):
        model.forward(np.array([[0, 6]]))  # 6 == vocab_size


def test_forward_rejects_negative_ids():
    """Negative ids used to silently wrap to the wrong embedding rows."""
    model = small_model()
    with pytest.raises(ValueError, match="token ids"):
        model.forward(np.array([[-1, 0]]))


def test_generate_rejects_out_of_vocab_prompt():
    model = small_model()
    with pytest.raises(ValueError, match="token ids"):
        model.generate(np.array([0, 99]), max_new=1)


def test_generate_negative_temperature_raises():
    """temperature=-1 used to be silently clamped to near-greedy."""
    model = small_model()
    with pytest.raises(ValueError, match="temperature"):
        model.generate(np.array([1, 2]), max_new=1, temperature=-1.0)


def test_generate_temperature_zero_is_greedy_and_deterministic():
    model = small_model()
    prompt = np.array([1, 2, 3])
    out1 = model.generate(prompt, max_new=6, temperature=0.0,
                          rng=np.random.default_rng(1))
    out2 = model.generate(prompt, max_new=6, temperature=0.0,
                          rng=np.random.default_rng(2))
    np.testing.assert_array_equal(out1, out2)
    # ... and it really is the argmax path
    logits, _ = model.forward(prompt[None, :])
    assert out1[3] == int(np.argmax(logits[0, -1]))


def test_generate_single_token_prompt():
    model = small_model()
    out = model.generate(np.array([1]), max_new=4, temperature=1.0,
                         rng=np.random.default_rng(0))
    assert out.shape == (5,)
    assert ((out >= 0) & (out < 6)).all()
