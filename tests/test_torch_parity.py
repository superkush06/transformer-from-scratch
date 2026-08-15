"""Every gradient against torch's autograd, across shapes the notebook never ran.

The notebook rebuilds this GPT with torch ops, copies the weights across, and
diffs every parameter gradient; `test_notebook.py` holds its prose to that
measurement. But the notebook audits one batch and four architectures, all at
`max_T`. This module reuses the notebook's own `mirror` and sweeps it over
configurations chosen to catch what a fixed example cannot: a single block, an
odd head count, a batch of one, a sequence shorter than the position table,
and a vocabulary bigger than the model width.

CI installs torch for exactly one job (`parity` in ci.yml), so this comparison
runs on every push instead of wherever torch happens to be installed. The
finite-difference suite says the backward pass agrees with the math; this says
it agrees with the other implementation of the math that the rest of the world
uses.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not a dependency of this package")

from tfs import GPT  # noqa: E402

# tests/ is not a package, so the notebook-runner helpers in test_notebook.py
# are loaded by path. The alias keeps pytest from collecting the file twice.
_spec = importlib.util.spec_from_file_location(
    "audit_notebook_runner", pathlib.Path(__file__).with_name("test_notebook.py"))
test_notebook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(test_notebook)

# (d_model, n_heads, n_blocks, d_ff, vocab, B, T, max_T, seed)
CONFIGS = [
    pytest.param(8, 2, 1, 16, 7, 2, 5, 6, 0, id="one-block"),
    pytest.param(16, 4, 2, 32, 11, 2, 6, 6, 1, id="notebook-shaped"),
    pytest.param(24, 3, 3, 48, 13, 1, 4, 8, 2, id="odd-heads-short-T"),
    pytest.param(32, 4, 2, 64, 29, 3, 7, 9, 3, id="wide-vocab-batch-3"),
    pytest.param(12, 6, 2, 20, 5, 2, 3, 6, 4, id="head-dim-2"),
]

# The notebook measures ~1e-15 in float64. Three decades of slack keeps a BLAS
# difference from failing the job while a real backward-pass regression, which
# shows up decades higher, still does. Same policy as test_notebook.py.
TOL = 1e-12


@pytest.fixture(scope="module")
def mirror():
    return test_notebook.run_upto("def mirror(")["mirror"]


@pytest.mark.parametrize("d_model,n_heads,n_blocks,d_ff,vocab,B,T,max_T,seed", CONFIGS)
def test_every_gradient_matches_torch(mirror, d_model, n_heads, n_blocks, d_ff,
                                      vocab, B, T, max_T, seed):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=(B, T))
    tgt = rng.integers(0, vocab, size=(B, T))

    model = GPT(vocab_size=vocab, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                n_blocks=n_blocks, max_T=max_T, seed=seed)
    for p in model.params():
        p.zero_grad()

    mine = model.loss_and_grads(ids, tgt)
    theirs, grads = mirror(model, n_heads, ids, tgt)

    assert mine == pytest.approx(theirs, rel=TOL)
    for name, p in model.named_params():
        ref = grads[name]
        worst = np.abs(p.grad - ref).max() / max(np.abs(ref).max(), 1e-12)
        assert worst < TOL, f"{name}: relative gradient error {worst:.2e} against torch"


def test_the_mirror_is_not_trivially_agreeing(mirror):
    """Break one weight after the copy and demand the comparison notices.

    A parity check that cannot fail measures nothing. Perturbing a weight
    AFTER the mirror has copied it makes the two models genuinely different,
    so the losses must part ways; if they do not, the mirror is reading the
    same buffers rather than holding its own.
    """
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 7, size=(2, 5))
    tgt = rng.integers(0, 7, size=(2, 5))
    model = GPT(vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_blocks=1,
                max_T=6, seed=0)
    for p in model.params():
        p.zero_grad()
    theirs, _ = mirror(model, 2, ids, tgt)
    # One entry, not a blanket `+= c`: adding the same constant to every column
    # of the output projection shifts each token's logits uniformly, and
    # softmax cannot see a uniform shift, so cross-entropy would not move.
    dict(model.named_params())["lm_head.W"].data[0, 0] += 0.25
    mine = model.loss_and_grads(ids, tgt)
    assert abs(mine - theirs) > 1e-6, "perturbing a weight did not move the comparison"
