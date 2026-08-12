"""The attention animation, checked as an artefact rather than as a picture.

`examples/make_attention_anim.py` trains a model for 3,000 steps, so the suite
cannot redraw the figure on every push. What it can do is hold the two
committed files to the constraints that make them work at all, every one of
which has a way of breaking silently:

* **size.** A README that pulls half a megabyte of SVG is a worse README.
  60 KB per file is the budget, and the quantisation in the generator exists
  to meet it.
* **self-containment.** GitHub serves these under
  `default-src 'none'; style-src 'unsafe-inline'; sandbox`, so a script, an
  external font, a `data:` URI or an `xlink:href` is not a degraded figure,
  it is a blank one.
* **one name, one meaning.** Every trajectory in the figure is a CSS
  animation, and the names are generated. A generated name that collides with
  a hand-written one silently drives the wrong element: an earlier draft
  animated the step labels with a cell's opacity curve and showed two
  checkpoints at once.
* **the causal mask.** The point of the upper triangle is that it is empty.
  A cell drawn above the diagonal would be a claim that position i read
  position j > i, and no amount of prose in the caption would outrank it.
* **every keyframe sets an opacity.** A value a browser rejects is a
  declaration a browser drops, and an empty keyframe is not the same as one
  holding zero: the animation loses that stop, and if the lost stop was the
  last one it falls back to the element's own opacity of 1. The invisible
  level shipped once spelled as a bare dot, which turned 122 cells whose
  weight had gone to nothing into the brightest cells in the hold.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FILES = [DOCS / "attention_light.svg", DOCS / "attention_dark.svg"]
BUDGET = 60 * 1024

sys.path.insert(0, str(ROOT / "examples"))
import make_attention_anim as gen  # noqa: E402

_HEX = re.compile(r"#[0-9A-Fa-f]{6}")
# CSS <number>: a dot needs a digit on one side of it, which is the whole point
# of this pattern being here.
_CSS_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)")
_HEAD = re.compile(r'<g fill="none" stroke-width="\d+" stroke="#[0-9A-Fa-f]{6}">'
                   r"(.*?)</g>", re.S)
_PATH = re.compile(r"<path(.*?)/>", re.S)
_MOVE = re.compile(r"([MmhH])([-\d. ]+)")


@pytest.fixture(scope="module", params=[p.name for p in FILES])
def svg(request) -> str:
    return (DOCS / request.param).read_text()


def segments(d: str) -> list[tuple[float, float, float]]:
    """(y, x, length) of every horizontal run in a path, following the pen.

    Cells are stroked segments rather than filled rectangles, which is how the
    file fits, so reading the geometry back means walking the path.
    """
    out, x, y = [], 0.0, 0.0
    for cmd, args in _MOVE.findall(d):
        nums = [float(v) for v in args.split()]
        if cmd == "M":
            x, y = nums
        elif cmd == "m":
            x, y = x + nums[0], y + nums[1]
        elif cmd in "hH":
            length = nums[0] if cmd == "h" else nums[0] - x
            out.append((y, x, length))
            x += length
    return out


def test_the_generator_and_the_files_agree_on_the_run():
    """The figure claims a shape; the script that drew it has to still say so."""
    light = FILES[0].read_text()
    phases = {int(m) for m in re.findall(r"@keyframes ph(\d+)", light)}
    assert phases == set(range(len(gen.CHECKPOINTS)))
    for step in gen.CHECKPOINTS:
        assert f">{step:,}</text>" in light, f"no label for step {step}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_each_file_fits_the_readme_budget(path):
    size = path.stat().st_size
    assert size <= BUDGET, f"{path.name} is {size:,} bytes, budget {BUDGET:,}"


def test_nothing_is_fetched_from_outside_the_file(svg):
    """The only URL allowed is the SVG namespace, which is a name, not a fetch."""
    for pattern in ("<script", "data:", "xlink:", "@import", "url("):
        assert pattern not in svg, f"{pattern} will not survive GitHub's CSP"
    assert re.findall(r"https?://\S+?[\"')]", svg) == ['http://www.w3.org/2000/svg"']


def test_every_animation_name_is_defined_exactly_once(svg):
    names = re.findall(r"@keyframes ([A-Za-z0-9]+)", svg)
    assert len(names) == len(set(names)), (
        "two @keyframes share a name, so one of them is dead: "
        f"{sorted({n for n in names if names.count(n) > 1})}")


def test_every_class_the_document_uses_is_styled(svg):
    used = {c for attr in re.findall(r'class="([^"]+)"', svg) for c in attr.split()}
    styled = set(re.findall(r"\.([A-Za-z][A-Za-z0-9]*)\{", svg))
    assert used <= styled, f"unstyled classes: {sorted(used - styled)}"


def test_every_animation_that_is_named_is_also_used(svg):
    """A keyframes block nobody references is 100-odd bytes of the budget."""
    used = {c for attr in re.findall(r'class="([^"]+)"', svg) for c in attr.split()}
    for name in re.findall(r"@keyframes ([A-Za-z0-9]+)", svg):
        assert name in used, f"@keyframes {name} is never applied"


def test_the_loop_is_the_right_length_and_ends_on_a_still(svg):
    """8 to 14 seconds, and the last checkpoint holds long enough to read."""
    durations = {float(d) for d in re.findall(r"animation-duration:([\d.]+)s", svg)}
    durations |= {float(d) for d in re.findall(r'dur="([\d.]+)s"', svg)}
    assert durations == {float(gen.DUR)}
    assert 8 <= gen.DUR <= 14
    last = len(gen.CHECKPOINTS) - 1
    body = re.search(rf"@keyframes ph{last}\{{(.*?)\}}\}}", svg).group(1)
    hold = int(re.search(r"(\d+)%,100%\{opacity:1\}", body + "}").group(1))
    assert (100 - hold) / 100 * gen.DUR >= 2.0, "the final frame flashes past"


def test_no_cell_is_ever_drawn_above_the_diagonal(svg):
    """Position i may only read positions up to i, in every frame."""
    heads = _HEAD.findall(svg)
    assert len(heads) == gen.ARCH["n_blocks"] * gen.ARCH["n_heads"]
    drawn = 0
    for body in heads:
        for attrs in _PATH.findall(body):
            if "stroke=" in attrs:            # the dimmed mask, checked below
                continue
            for y, x, length in segments(re.search(r'd="([^"]+)"', attrs).group(1)):
                row = (y - gen.CELL / 2) / gen.PITCH
                col = x / gen.PITCH
                assert row == int(row) and col == int(col), (row, col)
                assert length == gen.CELL, f"a cell is {length} wide"
                assert col <= row, f"cell (row {row}, column {col}) reads the future"
                drawn += 1
    expected = len(heads) * gen.WINDOW * (gen.WINDOW + 1) // 2
    assert drawn <= expected
    # A weight can quantise to invisible, so some cells are legitimately not
    # drawn. Most of them are, though, and a figure missing half its cells
    # would otherwise pass everything above.
    assert drawn > 0.6 * expected


def test_the_mask_is_drawn_over_the_whole_upper_triangle(svg):
    right = (gen.WINDOW - 1) * gen.PITCH + gen.CELL
    for body in _HEAD.findall(svg):
        attrs = next(a for a in _PATH.findall(body) if "stroke=" in a)
        bands = segments(re.search(r'd="([^"]+)"', attrs).group(1))
        assert len(bands) == gen.WINDOW - 1
        for i, (y, x, length) in enumerate(bands):
            assert (y - gen.CELL / 2) / gen.PITCH == i
            assert x == (i + 1) * gen.PITCH, "the mask starts off the diagonal"
            assert x + length == right, "the mask stops short of the edge"


def test_the_two_themes_are_the_same_figure_in_different_colours():
    stripped = [_HEX.sub("#", p.read_text()) for p in FILES]
    assert stripped[0] == stripped[1]
    palettes = [set(_HEX.findall(p.read_text())) for p in FILES]
    assert palettes[0] & palettes[1] == set(), "a colour is shared by both themes"


def test_the_opacity_ramp_is_monotone_and_spans_the_range():
    levels = [gen.level(w / 100) for w in range(101)]
    assert levels[0] == 0 and levels[-1] == gen.LEVELS
    assert levels == sorted(levels)
    weights = [gen.level_weight(k) for k in range(gen.LEVELS + 1)]
    assert weights == sorted(weights)
    # the legend prints the weight each level stands for, so the round trip
    # through the gamma has to land back on the level it names
    assert [gen.level(w) for w in weights] == list(range(gen.LEVELS + 1))


def test_every_level_spells_itself_as_a_css_number():
    """A shortened number is still a number, and "." is not one.

    `opacity` trims "0.00" at both ends, and the invisible level is the one
    where that leaves nothing to trim to. It shipped once as a bare dot.
    """
    for k in range(gen.LEVELS + 1):
        spelling = gen.opacity(k)
        assert _CSS_NUMBER.fullmatch(spelling), f"level {k} spelled {spelling!r}"
        assert float(spelling) == pytest.approx(k / gen.LEVELS)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_keyframe_leaves_its_opacity_out(path):
    """Every stop of every animation has to actually set an opacity.

    A declaration a browser rejects is dropped, and the keyframe holding it
    goes empty. That is not the same as a keyframe holding opacity 0: with no
    valid stop at 100% the animation falls back to the element's own opacity of
    1, so a cell whose weight vanished by the last checkpoint fades *up* to
    solid across the hold, and the frame a reader arrives on says the reverse
    of the run that drew it.
    """
    svg = path.read_text()
    for name, body in re.findall(r"@keyframes ([A-Za-z0-9]+)\{(.*?)\}\}", svg):
        stops = re.findall(r"((?:\d+%,)*\d+%)\{([^}]*)", body + "}")
        assert stops, f"@keyframes {name} has no stops"
        for at, decls in stops:
            value = re.fullmatch(r"opacity:(.*)", decls)
            assert value, f"@keyframes {name} sets nothing at {at}: {decls!r}"
            assert _CSS_NUMBER.fullmatch(value.group(1)), (
                f"@keyframes {name} at {at} is not a CSS number: "
                f"{value.group(1)!r}")
    for value in re.findall(r'opacity="([^"]*)"', svg):
        assert _CSS_NUMBER.fullmatch(value), f'opacity="{value}" is not a number'
