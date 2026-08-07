"""The Colab notebook is a fifth place a number can go stale.

`notebooks/audit.ipynb` makes claims in its prose and backs them with cells
whose output nobody re-runs. The README already refuses to let prose drift
from code; this does the same for the notebook, by executing its cells and
asserting the claims still hold.

torch is not a dependency of this package and CI does not install it, so the
cell that needs it is skipped there and exercised wherever it is available.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import re

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "audit.ipynb"


def cells():
    doc = json.loads(NB.read_text())
    return doc["cells"]


def code_cells():
    """Source of every code cell, with Colab's shell lines stripped."""
    out = []
    for i, c in enumerate(cells()):
        if c["cell_type"] != "code":
            continue
        src = "\n".join(line for line in "".join(c["source"]).splitlines()
                        if not line.lstrip().startswith("!"))
        out.append((i, src))
    return out


def run_upto(marker: str) -> dict:
    """Execute the notebook's cells in order, stopping after `marker` appears.

    Cells share one namespace in a notebook, so running one in isolation is
    not the same test. This reproduces the order a reader gets.
    """
    ns: dict = {}
    for i, src in code_cells():
        if "import torch" in src and _no_torch():
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"{NB.name}:cell{i}", "exec"), ns)
        if marker in src:
            return ns
    raise AssertionError(f"no cell containing {marker!r}")


def prose():
    return "\n".join("".join(c["source"]) for c in cells()
                     if c["cell_type"] == "markdown")


def test_the_notebook_is_valid_and_has_no_saved_output():
    """Committed outputs go stale silently and bloat the diff."""
    doc = json.loads(NB.read_text())
    assert doc["nbformat"] == 4
    for c in doc["cells"]:
        if c["cell_type"] == "code":
            assert c.get("outputs") == [], "clear outputs before committing"
            assert c.get("execution_count") is None


def test_every_code_cell_runs():
    """Execute the notebook end to end in one namespace, as Colab would."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    ns: dict = {}
    for i, src in code_cells():
        if "import torch" in src and _no_torch():
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(src, f"{NB.name}:cell{i}", "exec"), ns)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"cell {i} raised {type(exc).__name__}: {exc}")


def _no_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return True
    return False


def test_the_audit_helper_agrees_with_the_packaged_one():
    """The notebook defines its own `audit`; it must measure the same thing."""
    ns = run_upto("def audit(")
    from tfs import GPT
    model = GPT(vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_blocks=2,
                max_T=6, seed=0)
    n, worst, _ = ns["audit"](model, ns["IDS"], ns["TGT"])
    assert n == 1312, f"notebook audited {n} scalars, expected 1,312"
    assert worst < 1e-4, f"the derivation should pass its own audit, got {worst:.2e}"


def test_the_sabotage_cell_is_caught():
    """Section 3 invites the reader to break LayerNorm. Prove the check bites."""
    import tfs.layers as layers
    import tfs.ops as ops
    from tfs import GPT

    ns = run_upto("def audit(")
    true_backward = ops.layernorm_backward

    def broken(d_out, cache):
        x_hat, gamma, inv = cache
        N = x_hat.shape[-1]
        d_gamma = (d_out * x_hat).sum(axis=tuple(range(d_out.ndim - 1)))
        d_beta = d_out.sum(axis=tuple(range(d_out.ndim - 1)))
        d_x_hat = d_out * gamma
        d_x = (1.0 / N) * inv * (N * d_x_hat - d_x_hat.sum(axis=-1, keepdims=True))
        return d_x, d_gamma, d_beta

    try:
        for mod in (ops, layers):
            if hasattr(mod, "layernorm_backward"):
                mod.layernorm_backward = broken
        model = GPT(vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_blocks=2,
                    max_T=6, seed=0)
        _, worst, _ = ns["audit"](model, ns["IDS"], ns["TGT"])
        assert worst > 1.0, (
            f"dropping the covariance term should be caught loudly, got {worst:.2e}")
    finally:
        for mod in (ops, layers):
            if hasattr(mod, "layernorm_backward"):
                mod.layernorm_backward = true_backward


@pytest.mark.skipif(_no_torch(), reason="torch is not a dependency of this package")
def test_the_torch_claim_in_the_prose_still_holds():
    """The notebook says the two agree to about 1e-15. Check that number.

    This is the strongest claim the notebook makes, and the one a reader is
    most likely to try to reproduce, so it should not be allowed to drift.
    """
    ns = run_upto("def mirror(")
    from tfs import GPT
    model = GPT(vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_blocks=2,
                max_T=6, seed=0)
    for p in model.params():
        p.zero_grad()
    mine = model.loss_and_grads(ns["IDS"], ns["TGT"])
    theirs, grads = ns["mirror"](model, 2, ns["IDS"], ns["TGT"])
    assert mine == pytest.approx(theirs, rel=1e-12)
    worst = max(np.abs(p.grad - grads[n]).max() / max(np.abs(grads[n]).max(), 1e-12)
                for n, p in model.named_params())
    # The prose quotes a precision. Find the tightest one it claims and hold
    # the measurement to it, with three decades of slack so a BLAS difference
    # is not a failure while a real regression still is.
    claimed = [int(m) for m in re.findall(r"`1e-(\d\d)`", prose())]
    assert claimed, "the notebook should state the precision it achieves"
    tightest = max(claimed)
    assert worst < 10 ** (-(tightest - 3)), (
        f"notebook claims 1e-{tightest}; measured {worst:.2e}")


def test_the_notebook_links_resolve_to_things_in_this_repo():
    text = prose()
    for rel in re.findall(r"github\.com/superkush06/transformer-from-scratch/"
                          r"blob/main/([\w./-]+)", text):
        assert (ROOT / rel).exists(), f"notebook links to missing {rel}"
    assert "notebooks/audit.ipynb" in text, "the Colab badge should point at this file"
