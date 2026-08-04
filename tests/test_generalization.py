"""The flagship demo must generalise past its training window.

A model trained on a single fixed window memorises absolute-position ->
token mappings and degenerates (…,5,4,5,4,…) once the sliding context
window changes phase at position 8. Random-offset training has to keep
greedy generation exact far beyond that point.
"""

import numpy as np

from examples.train_pattern import PERIOD, train


def test_greedy_generation_reproduces_pattern_past_training_window():
    model = train(steps=300, log=None)
    prompt = np.array([1, 2, 3])
    # 25 new tokens: positions 3..27, i.e. 20 tokens past the max_T=8
    # training window, so the context has slid through every phase.
    out = model.generate(prompt, max_new=25, temperature=0.0)
    expected = [PERIOD[i % len(PERIOD)] for i in range(len(out))]
    assert out.tolist() == expected
