"""The README, executed.

Prose rots quietly: a number gets edited by hand, the code moves, and the
two drift apart with nothing failing. Everything a reader of the README is
invited to *run* is run here instead of trusted.

Three things are pinned:

* the interpreter session under "The API in one page" — every ``pycon``
  block in the README is replayed in order, sharing one namespace, and
  every line of output is diffed against what the file claims;
* the decode-work table — the integers in it come from
  ``docs.figures.decode_work``, which counts token-positions through the
  block stack, so a table row that no longer matches the counter fails;
* the two head-line counts, 1,312 scalars in 29 tensors, which are read
  off the model ``examples/gradcheck.py`` actually builds.

Floats are compared to a relative tolerance rather than character by
character (see ``RTOL``): the same expression can print a different last
digit under a different BLAS, and that is not a regression. Agreeing to
twelve significant figures is still far tighter than the slack a
different BLAS needs (a few ulps, ~1e-16 relative) and tight enough that
retyping the demo loss ``2.2640319260382604`` as ``2.2640319270382604``
— a change in the tenth digit — fails here.
"""

from __future__ import annotations

import doctest
import math
import re
from pathlib import Path

from docs.figures import decode_work
from examples.gradcheck import build_model

README = Path(__file__).resolve().parents[1] / "README.md"

#: Relative tolerance for numbers that appear in README output.
RTOL = 1e-12

_PYCON = re.compile(r"^```pycon\n(.*?)^```", re.M | re.S)
_NUMBER = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")
_TABLE_ROW = re.compile(
    r"^\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d.]+)x\s*\|$", re.M
)


class _NumericChecker(doctest.OutputChecker):
    """Exact match, or an identical skeleton with numbers close to RTOL."""

    def check_output(self, want: str, got: str, optionflags: int) -> bool:
        if super().check_output(want, got, optionflags):
            return True
        if [p.split() for p in _NUMBER.split(want)] != [
            p.split() for p in _NUMBER.split(got)
        ]:
            return False
        wanted, gotten = _NUMBER.findall(want), _NUMBER.findall(got)
        return len(wanted) == len(gotten) and all(
            math.isclose(float(a), float(b), rel_tol=RTOL, abs_tol=0.0)
            for a, b in zip(wanted, gotten, strict=True)
        )


def _pycon_blocks() -> list[tuple[int, str]]:
    """Every ``pycon`` block in the README, with its 1-based line number."""
    text = README.read_text()
    return [
        (text.count("\n", 0, m.start()) + 1, m.group(1))
        for m in _PYCON.finditer(text)
    ]


def test_readme_exposes_the_blocks_this_module_claims_to_check():
    # Without this, a fence renamed in the README would leave the doctest
    # below iterating over nothing and passing for free.
    blocks = _pycon_blocks()
    assert len(blocks) == 2
    parser = doctest.DocTestParser()
    examples = sum(
        len([e for e in parser.parse(src) if isinstance(e, doctest.Example)])
        for _, src in blocks
    )
    assert examples >= 15


def test_readme_session_reproduces_its_own_output():
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(
        checker=_NumericChecker(),
        optionflags=doctest.NORMALIZE_WHITESPACE,
        verbose=False,
    )
    report: list[str] = []
    globs: dict = {}
    for lineno, source in _pycon_blocks():
        test = parser.get_doctest(
            source, globs, f"README.md:{lineno}", str(README), lineno
        )
        runner.run(test, out=report.append, clear_globs=False)
        globs = test.globs
    assert runner.failures == 0, "".join(report)


def test_readme_decode_table_matches_the_counted_positions():
    rows = _TABLE_ROW.findall(README.read_text())
    assert len(rows) == 5, "the decode-work table lost or gained rows"
    lengths, full, cached = decode_work()
    assert [int(r[0].replace(",", "")) for r in rows] == list(lengths)
    assert [int(r[1].replace(",", "")) for r in rows] == list(full)
    assert [int(r[2].replace(",", "")) for r in rows] == list(cached)
    assert [r[3] for r in rows] == [f"{f / c:.1f}" for f, c in zip(full, cached, strict=True)]


def test_readme_gradcheck_counts_come_from_the_model():
    model = build_model()
    scalars = sum(p.data.size for _, p in model.named_params())
    tensors = len(model.named_params())
    assert (scalars, tensors) == (1312, 29)
    text = README.read_text()
    # These are counts of derivatives and of tensors. The number of tests
    # is what `pytest` prints, and it is a different number.
    assert f"{scalars:,} gradient checks" in text
    assert f"{tensors} parameter tensors" in text
