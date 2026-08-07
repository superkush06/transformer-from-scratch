"""The browser demo ships a copy of the package, so it can fall behind.

`docs/demo/` serves `tfs` as a zip that Pyodide unpacks. A copy drifts: change
a backward pass, forget to rebuild, and the page keeps auditing last month's
derivation while every number on it cites this month's. These tests fail
instead of letting that happen quietly.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"
BUNDLE = DEMO / "tfs-pkg.zip"
DRIVER = DEMO / "tfsdemo.py"
REBUILD = "stale; run `python docs/demo/make_pkg.py`"

sys.path.insert(0, str(DEMO))
from make_pkg import sources  # noqa: E402


@pytest.mark.parametrize("path", [p.name for p in sources()])
def test_bundled_source_is_byte_identical(path):
    """Every module in the zip must match the one on disk."""
    src = next(p for p in sources() if p.name == path)
    with zipfile.ZipFile(BUNDLE) as z:
        shipped = z.read(str(src.relative_to(ROOT)))
    assert shipped == src.read_bytes(), f"{path} is {REBUILD}"


def test_the_bundle_holds_every_module():
    with zipfile.ZipFile(BUNDLE) as z:
        names = set(z.namelist())
    want = {str(p.relative_to(ROOT)) for p in sources()}
    assert names == want, REBUILD


def test_stamp_matches_both_assets():
    """The page hangs cache-busting versions off these hashes.

    The zip and the driver are fetched separately and change independently,
    so each needs its own. Sharing one means a driver-only edit bumps nothing
    and a CDN keeps serving the previous deploy's `tfsdemo.py` against fresh
    HTML.
    """
    stamp = json.loads((DEMO / "bundle.json").read_text())
    assert stamp["sha"] == hashlib.sha256(BUNDLE.read_bytes()).hexdigest()[:12], REBUILD
    assert stamp["driver"] == hashlib.sha256(DRIVER.read_bytes()).hexdigest()[:12], REBUILD
    assert stamp["modules"] == len(sources()), REBUILD


def test_the_driver_imports_only_what_pyodide_will_have():
    tree = ast.parse(DRIVER.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
    allowed = set(sys.stdlib_module_names) | {"tfs", "numpy", "__future__"}
    missing = sorted(imported - allowed)
    assert not missing, f"tfsdemo.py imports {missing}, which Pyodide will not have"


def test_the_page_fetches_the_versioned_urls():
    """A stale asset against fresh HTML is an ImportError with no clue in it."""
    html = (DEMO / "index.html").read_text()
    for asset in ("tfs-pkg.zip?v=${stamp.sha}", "tfsdemo.py?v=${stamp.driver}"):
        assert asset in html, f"index.html should fetch {asset}"


def test_every_sabotage_is_caught_by_the_audit():
    """The page's central claim, asserted here rather than only on the page.

    Each wrong derivation has to be invisible to the forward pass and loud to
    the gradient check. If a bug ever stopped failing, panel 03 would quietly
    become a page that breaks something and reports that nothing broke.
    """
    import tfsdemo as d

    def worst():
        d.audit_begin()
        while True:
            s = d.audit_step(700)
            if s["done"]:
                return s

    d.reset_batch()
    d.set_bug("none")
    clean = worst()
    assert clean["failing"] == [], "the derivation as written must pass"
    assert clean["worst"] < 1e-4

    for bug in ("ln_cov", "ln_mean", "gelu"):
        d.set_bug(bug)
        got = worst()
        assert got["worst"] > 1.0, f"{bug} should be caught loudly, got {got['worst']:.2e}"
        assert len(got["failing"]) >= 10, f"{bug} should contaminate many tensors"
        # and some tensor must survive, because a gradient that never flows
        # through the broken op is still exactly right. That contrast is the
        # point of the panel.
        assert len(got["failing"]) < 29, f"{bug} should leave some tensors clean"
    d.set_bug("none")


def test_a_readers_batch_changes_the_numbers_but_not_the_verdict():
    """Panel 02's claim: whatever data you give it, the derivatives agree."""
    import tfsdemo as d

    d.set_bug("none")
    base = d.reset_batch()
    other = d.set_batch([[6, 5, 4, 3, 2, 1, 0], [0, 0, 6, 6, 1]])
    assert other["loss"] != pytest.approx(base["loss"], rel=1e-6), \
        "a different batch should give a different loss"

    d.audit_begin()
    while True:
        s = d.audit_step(700)
        if s["done"]:
            break
    assert s["failing"] == [], "the audit must hold on a reader-chosen batch"
    assert s["worst"] < 1e-4
    d.reset_batch()
