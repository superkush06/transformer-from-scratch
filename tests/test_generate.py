"""Input-validation edge cases for GPT.forward and generate."""

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
