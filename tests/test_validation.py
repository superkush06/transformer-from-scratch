"""CI enforcement for the validation ladder in docs/validation.md.

`examples/validate.py` is the thing you run to get the numbers the doc
quotes; this module runs the same checks on smaller budgets so every one
of them is pinned on every push. If a rung here fails, the corresponding
row of the doc is no longer true.

The two rungs that disagree with their reference on purpose — the tanh
GELU and LayerNorm's variance floor — are asserted to keep disagreeing
by the documented amount. A "fix" that silently made either of them
agree would change the library's behaviour, and the doc would be stale
rather than wrong.
"""

import numpy as np

from examples.regime_handoff import (
    bigram_cross_entropy,
    entropy_rates,
    fit,
    model_cross_entropy,
    oracle_cross_entropy,
    simulate,
    transition_tensor,
)
from examples.validate import (
    check_adam_first_step,
    check_attention_reference,
    check_causal_mask,
    check_gelu_against_exact,
    check_gradients,
    check_layernorm_variance,
    check_overfit_one_batch,
    check_parameter_count,
    check_score_scaling,
    check_uniform_cross_entropy,
)


def test_every_parameter_gradient_agrees_with_central_differences():
    assert check_gradients().agrees


def test_attention_matches_a_loop_implementation_of_the_paper():
    """The vectorised path and a naive triple loop are the same function."""
    assert check_attention_reference().agrees


def test_causal_mask_admits_no_future_information():
    assert check_causal_mask(trials_per_position=8).agrees


def test_the_model_can_memorise_one_batch():
    """Karpathy's rung: a model that cannot overfit one batch is broken.

    Fewer steps than `examples/validate.py` uses, so the threshold is
    looser — but the accuracy requirement is not.
    """
    check = check_overfit_one_batch(steps=1500)
    loss, acc = check.ours.split(" / ")
    assert float(loss) < 1e-4, check.ours
    assert float(acc) == 1.0, check.ours


def test_dot_product_variance_is_the_head_dimension():
    assert check_score_scaling(d_k=64, n=50_000).agrees


def test_closed_form_rungs_are_exact():
    assert check_uniform_cross_entropy().agrees
    assert check_parameter_count().agrees
    assert check_adam_first_step().agrees


def test_the_documented_disagreements_still_disagree():
    """Rows 6 and 8 of docs/validation.md must stay honest.

    The tanh GELU differs from the exact x·Phi(x) by ~4.7e-04, and
    LayerNorm's output variance sits below one by the variance floor.
    Both are deliberate; both are stated in the doc as gaps rather than
    matches, so this test fails if either quietly becomes exact.
    """
    gelu = check_gelu_against_exact()
    assert not gelu.agrees
    assert 4e-4 < float(gelu.ours.split()[0]) < 6e-4, gelu.ours

    ln = check_layernorm_variance()
    assert not ln.agrees
    assert 0.99999 < float(ln.ours) < 1.0, ln.ours


def test_held_out_cross_entropy_sits_between_the_bigram_and_the_bayes_floor():
    """The information-theoretic sandwich, on a smaller training budget.

    A model that scored *below* the oracle by a wide margin would mean the
    evaluation had leaked; one that failed to beat the bigram would mean it
    had not learned the second-order structure at all. Both edges are
    asserted, with slack for the finite test draw.
    """
    P = transition_tensor()
    h2, h1 = entropy_rates(P)
    assert h1 > h2 > 0.0
    assert np.isclose(h1 - h2, 0.0260, atol=5e-4)  # I(X; X-2 | X-1)

    train_seq = simulate(P, 6_000, seed=1)
    test_seq = simulate(P, 1_500, seed=2)
    model = fit(train_seq, steps=700)
    ours = model_cross_entropy(model, test_seq)
    oracle = oracle_cross_entropy(P, test_seq)
    bigram = bigram_cross_entropy(train_seq, test_seq)

    assert ours < bigram, (ours, bigram)
    assert ours > oracle - 0.01, (ours, oracle)
    assert ours < oracle + 0.02, (ours, oracle)
