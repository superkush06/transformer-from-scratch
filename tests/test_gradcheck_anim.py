"""The gradient-audit animation, held to the file it claims to be.

`docs/gradcheck_light.svg` and its dark twin sit near the top of the README,
fetched from raw.githubusercontent.com under `default-src 'none'; style-src
'unsafe-inline'; sandbox`. Inline `@keyframes` runs there; a script, a webfont,
an external image and a `data:` URI do not, and none of them fails loudly. The
figure just stops moving, or comes back as a blank box, on the first page
anyone sees. So the first part of this module checks what the renderer cares
about rather than the picture, and the second part checks that it moves: that
the fill only ever grows, that one tensor is named at a time, and that the
verdict is not on screen while coordinates are still landing.

The last part is the one worth the most. The sweep costs about a quarter of a
second, so there is no excuse for trusting the figures printed in the frame:
`check_every_scalar()` runs here, and the dots in the file have to be that run
transformed by the generator's own axes, in order, all 1,288 of them. Every
number beside them, and every number in the README caption, is rebuilt from the
same arrays. The failure this exists to catch is a label that survived a change
to the model which moved the quantity under it.

The caption also makes a claim the plot cannot: that the 24 coordinates missing
from the log axes are *exactly* zero, not merely small. The generator only
knows they fall under 1e-8, so that sentence would still read plausibly if one
went tiny-but-nonzero. It is asserted directly instead.

Sister modules: `tests/test_attention_anim.py`, `tests/test_learning_anim.py`.
Regenerate with `PYTHONPATH=. python examples/make_gradcheck_anim.py`.
"""

from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from examples.gradcheck import TOLERANCE, check_every_scalar
from examples.make_gradcheck_anim import DOT, HI, LO, RESOLVED_FLOOR, X0, X1, sx, sy

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
README = ROOT / "README.md"
FIGURES = ("gradcheck_light.svg", "gradcheck_dark.svg")
SVGNS = "{http://www.w3.org/2000/svg}"

BUDGET = 60 * 1024        # what a README can afford to pull, per file
LOOP = (8.0, 14.0)        # long enough to read, short enough to sit through
MIN_HOLD = 1.5            # dead-still seconds at the end, so it reads as a still
GENERIC = {"serif", "sans-serif", "monospace"}

REGEN = "regenerate with `PYTHONPATH=. python examples/make_gradcheck_anim.py`"


@pytest.fixture(params=FIGURES)
def svg(request) -> str:
    return (DOCS / request.param).read_text()


@pytest.fixture(scope="module")
def sweep():
    """One exhaustive sweep, shared across this module."""
    return check_every_scalar()


@pytest.fixture(scope="module")
def facts(sweep):
    """Everything the frame and the caption assert, recomputed from the sweep."""
    names, analytic, numeric = sweep
    resolved = np.abs(numeric) > RESOLVED_FLOOR
    rel = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-30)
    worst = float(rel[resolved].max())
    return {
        "worst": worst,
        "where": str(names[resolved][int(np.argmax(rel[resolved]))]),
        "margin": TOLERANCE / worst,
        "scalars": len(names),
        "tensors": len(dict.fromkeys(names.tolist())),
        "drawn": int(resolved.sum()),
        "dropped": int((~resolved).sum()),
        "signs": int((np.sign(analytic) == np.sign(numeric)).sum()),
        "resolved": resolved,
        "order": list(dict.fromkeys(names.tolist())),
        "rel": rel,
    }


# --------------------------------------------------------------------------
# reading the file back
# --------------------------------------------------------------------------

def css(svg: str) -> str:
    return re.search(r"<style>(.*?)</style>", svg, re.S).group(1)


def keyframes(svg: str) -> dict[str, list[tuple[float, str]]]:
    """{name: [(offset percent, declarations)]}, in offset order."""
    out: dict[str, list[tuple[float, str]]] = {}
    for name, body in re.findall(r"@keyframes\s+([A-Za-z0-9_]+)\{(.*?)\}\}", css(svg), re.S):
        out[name] = sorted(
            (float(o.strip().rstrip("%")), decl)
            for sel, decl in re.findall(r"([0-9.,%\s]+)\{([^}]*)\}", body + "}")
            for o in sel.split(",") if o.strip()
        )
    return out


def duration(svg: str) -> float:
    return float(re.search(r"animation-duration:([\d.]+)s", css(svg)).group(1))


def opacity_at(stops: list[tuple[float, str]], t: float, total: float) -> float:
    """The opacity a browser shows at `t`, including the stop it invents.

    A keyframes block that stops short of 100% gets its final stop built from
    the element's own style. Every animated element here is opaque by default,
    so that synthesised stop is opacity 1, which is exactly how a batch of dots
    holds still after it lands.
    """
    pts = [(o, float(m.group(1)))
           for o, decl in stops
           if (m := re.search(r"(?<!stroke-)opacity:([0-9.]+)", decl))]
    assert pts, "this track does not animate opacity"
    if stops[-1][0] < 100.0:
        pts.append((100.0, 1.0))
    p = t / total * 100.0
    if p <= pts[0][0]:
        return pts[0][1]
    for (o1, v1), (o2, v2) in zip(pts, pts[1:], strict=False):
        if o1 <= p <= o2:
            return v1 if o2 == o1 else v1 + (v2 - v1) * (p - o1) / (o2 - o1)
    return pts[-1][1]


def tracks(svg: str, prefix: str) -> list[list[tuple[float, str]]]:
    """Every keyframes block named `<prefix><n>`, in numeric order."""
    kf = keyframes(svg)
    names = sorted((n for n in kf if re.fullmatch(prefix + r"\d+", n)),
                   key=lambda n: int(n[len(prefix):]))
    return [kf[n] for n in names]


def cues(stops: list[tuple[float, str]]) -> tuple[float, float]:
    """(offset it starts moving, offset it reaches full opacity), in percent.

    Every track here is invisible until some offset and opaque from another, so
    those two numbers are its whole schedule.
    """
    leaves = max(o for o, decl in stops if re.search(r"(?<!stroke-)opacity:0(?!\.)", decl))
    lands = min(o for o, decl in stops if re.search(r"(?<!stroke-)opacity:1\b", decl))
    return leaves, lands


def plotted(svg: str) -> list[tuple[int, int]]:
    """Every dot in the scatter, in the order the document places it.

    A coordinate is a near-zero-length subpath under `stroke-linecap:round`, so
    the dots of one batch live in one path's `d` as `M x y h.02` repeated.
    """
    out = []
    for d in re.findall(r'<path class="d a c\d+" d="([^"]*)"', svg):
        out += [(int(x), int(y)) for x, y in re.findall(r"M(-?\d+) (-?\d+)h", d)]
    return out


# --------------------------------------------------------------------------
# what GitHub's sandbox will and will not render
# --------------------------------------------------------------------------

def test_parses_as_xml(svg):
    ET.fromstring(svg)


def test_fits_the_readme_budget(svg):
    assert len(svg.encode()) < BUDGET, REGEN


def test_carries_no_script(svg):
    assert "<script" not in svg.lower()
    assert not re.search(r"\son\w+\s*=", svg), "no inline event handlers"


def test_fetches_nothing_from_outside(svg):
    """The only URL the file is allowed to name is the SVG namespace."""
    for ref in re.findall(r"url\(([^)]*)\)", svg) + re.findall(r'href="([^"]*)"', svg):
        assert ref.startswith("#"), f"external reference {ref!r}"
    assert "data:" not in svg
    assert "@font-face" not in svg
    assert "@import" not in svg
    assert "xlink" not in svg
    assert "<image" not in svg
    assert set(re.findall(r"https?://[^\"'\s>)]+", svg)) == {"http://www.w3.org/2000/svg"}


def test_every_text_element_resolves_to_a_font_stack(svg):
    """Resolve the family each `<text>` really gets, and require a generic.

    This file styles type once, with a `text{...}` rule, rather than repeating
    a `font-family` attribute 122 times — that rule is part of how it fits the
    budget. So the check walks each element's ancestors for an attribute and
    falls back to the rule, which keeps it honest whichever way a future
    generator chooses to say it.
    """
    root = ET.fromstring(svg)
    parents = {child: parent for parent in root.iter() for child in parent}
    rule = re.search(r"(?:^|[};])\s*text\{[^}]*font-family:([^;}]+)", css(svg))

    texts = list(root.iter(f"{SVGNS}text"))
    assert len(texts) > 50, "the figure lost its labels"
    for node in texts:
        family, hop = None, node
        while hop is not None and family is None:
            family = hop.get("font-family")
            hop = parents.get(hop)
        family = family or (rule.group(1) if rule else None)
        assert family, f"no font stack reaches {''.join(node.itertext())!r}"
        last = family.rsplit(",", 1)[-1].strip().strip("'\"")
        assert last in GENERIC, f"{family!r} ends in a font, not a generic family"


# --------------------------------------------------------------------------
# it moves, and every animation drives something
# --------------------------------------------------------------------------

def test_every_animated_element_names_a_keyframes_that_exists(svg):
    declared = set(keyframes(svg))
    applied = dict(re.findall(r"\.([A-Za-z0-9_]+)\{animation-name:([A-Za-z0-9_]+)\}", css(svg)))
    assert declared == set(applied.values()), \
        f"orphans: {sorted(declared ^ set(applied.values()))}; {REGEN}"
    assert len(re.findall(r"@keyframes ([A-Za-z0-9_]+)", css(svg))) == len(declared), \
        "two @keyframes share a name, so one of them is dead"

    animated = re.findall(r'class="([^"]*\ba\b[^"]*)"', svg)
    assert animated, "nothing carries the animation class"
    for classes in animated:
        assert set(classes.split()) & set(applied), \
            f'class="{classes}" is animated with no animation-name; {REGEN}'


def test_no_keyframes_is_a_no_op(svg):
    """An animation that sets nothing visible is budget spent on nothing."""
    for name, stops in keyframes(svg).items():
        touched = {prop for _, decl in stops for prop in re.findall(r"([a-z-]+):", decl)}
        assert touched & {"opacity", "stroke-width", "stroke-opacity"}, name
        assert len({decl for _, decl in stops}) > 1, f"@keyframes {name} never changes anything"


def test_keyframe_offsets_are_ordered_and_inside_the_loop(svg):
    for name, stops in keyframes(svg).items():
        offs = [o for o, _ in stops]
        assert offs == sorted(offs), name
        assert len(set(offs)) == len(offs), \
            f"@keyframes {name} repeats an offset, and Chrome merges those into a ramp"
        assert offs[0] >= 0.0 and offs[-1] <= 100.0, name


def test_the_loop_is_readable_and_ends_on_a_still(svg):
    total = duration(svg)
    assert len({float(d) for d in re.findall(r"animation-duration:([\d.]+)s", css(svg))}) == 1
    assert LOOP[0] <= total <= LOOP[1], f"the loop is {total}s; {REGEN}"
    assert "animation-iteration-count:infinite" in css(svg)

    moving = [o for stops in keyframes(svg).values() for o, _ in stops if o < 100.0]
    hold = total - max(moving) / 100.0 * total
    assert hold >= MIN_HOLD, \
        f"only {hold:.2f}s of the loop is still, so a reader arriving late sees motion"


def test_the_fill_only_ever_grows(svg):
    """Sampled across the loop, the plot and the bar never lose ground.

    This is what the caption promises — coordinates arriving in the order the
    sweep visits them — and it is what an editing mistake in the timing
    arithmetic breaks: a batch whose window opens early shows up here as a
    count that dips.
    """
    total = duration(svg)
    dots, bars = tracks(svg, "c"), tracks(svg, "b")
    assert (len(dots), len(bars)) == (69, 29), f"{len(dots)} batches, {len(bars)} tensors"

    counts = [(sum(opacity_at(s, total * i / 240, total) > 0.5 for s in dots),
               sum(opacity_at(s, total * i / 240, total) > 0.5 for s in bars))
              for i in range(241)]
    assert counts[0] == (0, 0), "the figure opens with points already placed"
    assert counts[-1] == (len(dots), len(bars)), "the figure never finishes filling"
    for (d0, b0), (d1, b1) in zip(counts, counts[1:], strict=False):
        assert d1 >= d0 and b1 >= b0, "the fill goes backwards"
    assert len({d for d, _ in counts}) > 40, "the fill arrives in too few visible steps"


def test_the_batches_land_in_the_order_the_sweep_visits_them(svg):
    """Batch i cannot arrive before batch i-1.

    The caption's claim is an order, and monotonicity of the total misses a
    violation of it: a batch scheduled at the top of the loop makes the count
    start at one rather than dip, so the count never goes backwards and the
    figure still lies about the order. Reading each track's own two cues is
    what pins it.
    """
    for prefix, what in (("c", "coordinate batches"), ("b", "swept slices of the bar")):
        schedule = [cues(s) for s in tracks(svg, prefix)]
        starts = [leaves for leaves, _ in schedule]
        lands = [land for _, land in schedule]
        assert starts == sorted(starts), f"the {what} do not start in order; {REGEN}"
        assert lands == sorted(lands), f"the {what} do not finish in order; {REGEN}"
        assert starts[0] > 0.0, f"the first of the {what} is already there at t=0"
        assert lands[-1] < 100.0, f"the last of the {what} never lands inside the loop"


def test_at_most_one_tensor_is_named_at_a_time(svg):
    """The panel names the tensor under the differences, and only that one."""
    total = duration(svg)
    wins = tracks(svg, "n")
    assert len(wins) == 29
    lit = [sum(opacity_at(s, total * i / 720, total) > 0.5 for s in wins) for i in range(721)]
    assert max(lit) == 1, "two tensor names are readable at once"
    assert sum(1 for n in lit if n == 1) > 500, "the name is absent for most of the loop"


def test_the_verdict_waits_for_the_last_coordinate(svg):
    """`PASS` must not be readable while coordinates are still landing.

    The two ramps do overlap by a few hundredths of a second by construction:
    the summary starts fading in as the last batch finishes settling. So this
    asks about readable rather than non-zero opacity, which is the claim a
    reader can actually be misled by.
    """
    total = duration(svg)
    summary, dots = keyframes(svg)["sm"], tracks(svg, "c")
    for i in range(241):
        t = total * i / 240
        if opacity_at(summary, t, total) > 0.5:
            assert sum(opacity_at(s, t, total) > 0.5 for s in dots) == len(dots), \
                f"the verdict is readable at t={t:.2f}s with coordinates still landing"


# --------------------------------------------------------------------------
# the two files are one figure
# --------------------------------------------------------------------------

def test_the_two_themes_are_the_same_figure_in_different_colours():
    files = {name: (DOCS / name).read_text() for name in FIGURES}
    roots = {name: re.search(r":root\{([^}]*)\}", text).group(1)
             for name, text in files.items()}
    bodies = {name: text.split("</style>")[1] for name, text in files.items()}
    rest = {name: css(text).split("}", 1)[1] for name, text in files.items()}

    assert len(set(bodies.values())) == 1, "the two themes draw different geometry"
    assert len(set(rest.values())) == 1, "the two themes animate differently"
    assert len(set(roots.values())) == 2, "the two themes carry the same palette"

    palettes = {name: dict(re.findall(r"--(\w+):(#[0-9A-Fa-f]{6})", block))
                for name, block in roots.items()}
    light, dark = (palettes[f"gradcheck_{t}.svg"] for t in ("light", "dark"))
    assert light.keys() == dark.keys() and len(light) == 7
    assert not set(light.values()) & set(dark.values()), "a colour is shared across themes"
    for name, text in files.items():
        assert set(re.findall(r"#[0-9A-Fa-f]{6}", text)) == set(palettes[name].values()), \
            f"{name} hard-codes a colour outside :root, so one theme will be wrong"


def test_the_backgrounds_are_the_two_grounds_the_repository_uses():
    """A figure with no ground of its own inherits GitHub's, and the paper
    ground is what the axis type was chosen against."""
    for theme, ground in (("light", "#F7F4EF"), ("dark", "#0D1117")):
        text = (DOCS / f"gradcheck_{theme}.svg").read_text()
        assert f"--bg:{ground}" in text
        assert '<rect width="900" height="572" fill="var(--bg)"/>' in text


# --------------------------------------------------------------------------
# every number in the frame, and in the caption, comes off the sweep
# --------------------------------------------------------------------------

def test_the_dots_are_the_sweep_put_through_the_generators_axes(sweep, facts):
    """The scatter has to be the run, coordinate for coordinate and in order.

    Nothing else in the suite would notice a figure redrawn from a different
    batch, a subsample, or a stale run: it would still be points on a diagonal.
    """
    names, analytic, numeric = sweep
    keep = facts["resolved"]
    want = list(zip(np.rint(sx(np.abs(numeric[keep])) * DOT).astype(int).tolist(),
                    np.rint(sy(np.abs(analytic[keep])) * DOT).astype(int).tolist(),
                    strict=True))
    for name in FIGURES:
        got = plotted((DOCS / name).read_text())
        assert len(got) == facts["drawn"], \
            f"{name} plots {len(got)} coordinates, the sweep resolves {facts['drawn']}"
        assert got == want, f"{name} is not this run; {REGEN}"
    assert len(names) == facts["scalars"]


def test_the_dropped_coordinates_are_exactly_zero_not_merely_small(sweep, facts):
    """The caption says exactly zero. The generator only knows they are tiny."""
    _, analytic, numeric = sweep
    drop = ~facts["resolved"]
    assert facts["dropped"] == 24
    assert not analytic[drop].any(), "a dropped coordinate has a non-zero gradient"
    assert not numeric[drop].any(), "a dropped coordinate has a non-zero difference"


def test_every_figure_printed_in_the_frame_comes_off_the_sweep(svg, facts):
    for claim in (
        f"{facts['scalars']:,} hand-derived gradients",
        f"{facts['tensors']} parameter tensors",
        f"{facts['worst']:.2e}",
        f"the loosest tensor is {facts['where']}",
        f"{facts['margin']:,.0f}x inside the tolerance",
        f"signs agree on {facts['signs']:,} of {facts['scalars']:,} coordinates",
        f"{facts['dropped']} of the {facts['scalars']:,} coordinates are exactly zero in both",
        f"the axes hold {facts['drawn']:,}",
        f"{facts['scalars']:,} of {facts['scalars']:,} scalars",
    ):
        assert claim in svg, f"the frame no longer says {claim!r}; {REGEN}"
    assert "PASS" in svg and facts["worst"] < TOLERANCE


def test_the_panel_counts_the_running_worst_and_not_the_final_one(svg, sweep, facts):
    """The 29 panels have to read as a running maximum over this run.

    Printing the finished number in every window would look identical on the
    last frame and be a lie on the other 28. Reading every window back is also
    what stops a partial edit — one occurrence of a number changed and the rest
    left alone — from slipping past a test that only asks whether the string
    appears somewhere.
    """
    names, _, _ = sweep
    windows = re.findall(r'<g class="a n\d+">(.*?)</g>', svg)
    assert len(windows) == facts["tensors"]

    running, best = [], 0.0
    for tensor in facts["order"]:
        here = facts["rel"][(names == tensor) & facts["resolved"]]
        best = max(best, float(here.max()))
        running.append(best)

    got = [re.findall(r'class="s19 ink">([^<]*)<', w) for w in windows]
    assert [pair[0] for pair in got] == facts["order"], \
        f"the panel names tensors in a different order than the sweep visits them; {REGEN}"

    # The early running maxima sit at ~1e-8, which is the summation-order noise
    # floor: ubuntu's BLAS and macOS's disagree there in the third digit, so
    # demanding the committed strings byte-equal a fresh sweep fails on
    # whichever OS didn't render the figure. What is stable everywhere, and
    # what a partial edit can't fake, is the shape: every window within a
    # factor of two of this run's running worst, never decreasing, ending on
    # the final worst — which lives two decades above the floor and formats
    # identically on both runners.
    shown = [float(pair[1]) for pair in got]
    assert all(b <= a for a, b in zip(shown[1:], shown[:-1])), \
        f"the panel numbers decrease somewhere, so they are not a running maximum; {REGEN}"
    for seen, ours in zip(shown, running):
        assert ours / 2 <= seen <= ours * 2, \
            f"a window reads {seen:.2e} where this run's running worst is {ours:.2e}; {REGEN}"
    assert got[-1][1] == f"{facts['worst']:.2e}", f"the last window is not this run's worst; {REGEN}"


def test_the_scalar_ranges_tile_the_sweep_without_a_gap(svg, sweep, facts):
    """`scalars A to B of N` has to account for every coordinate, once."""
    names, _, _ = sweep
    ranges = [tuple(int(v.replace(",", "")) for v in m)
              for m in re.findall(r"scalars ([\d,]+) to ([\d,]+) of ([\d,]+)", svg)]
    assert len(ranges) == facts["tensors"]
    assert {total for _, _, total in ranges} == {facts["scalars"]}

    expected = 1
    for (lo, hi, _), tensor in zip(ranges, facts["order"], strict=True):
        assert lo == expected, f"{tensor} starts at {lo}, not {expected}"
        assert hi - lo + 1 == int((names == tensor).sum()), f"{tensor} claims the wrong width"
        expected = hi + 1
    assert expected - 1 == facts["scalars"], "the windows do not reach the last scalar"


def test_the_footer_states_a_geometry_it_can_be_held_to(svg, facts):
    """How far off the line the worst point sits, in pixels, is checkable.

    It beats asserting that the points are "on" the diagonal, and it is the one
    sentence in the frame that depends on the plot's own scale, so it breaks if
    the axes are ever rescaled without redrawing the prose.
    """
    off = float(np.log10(1.0 + facts["worst"]) * (X1 - X0) / (HI - LO) * np.sqrt(0.5))
    assert (f"the worst disagreement is {facts['worst']:.2e} relative, which at this scale "
            f"sits {off:.0e} of a pixel off the line.") in svg, REGEN
    assert off < 1e-4, "the worst point is now far enough off the line to see"


def test_the_readme_caption_describes_this_figure(facts):
    """The caption is read before the picture by anyone who arrives mid-loop.

    It promises motion and names four quantities. All four are rebuilt from the
    sweep here, because a caption that disagrees with the figure above it is
    worse than no caption.
    """
    raw = README.read_text()
    flat = re.sub(r"\s+", " ", raw)

    assert re.search(r'<source media="\(prefers-color-scheme: dark\)" '
                     r'srcset="docs/gradcheck_dark\.svg">', raw), \
        "the dark file is no longer paired with prefers-color-scheme"
    assert 'src="docs/gradcheck_light.svg"' in raw
    alt = re.search(r'alt="([^"]*)" src="docs/gradcheck_light\.svg"', raw)
    assert alt and "filling in" in alt.group(1), \
        f"the alt text stopped describing the motion: {alt and alt.group(1)!r}"

    for claim in (
        f"{facts['drawn']:,} of the {facts['scalars']:,} coordinates land on",
        f"It ends at {facts['worst']:.2e}, in `{facts['where']}`, "
        f"which is {facts['margin']:,.0f}x inside",
        f"The other {facts['dropped']} coordinates are exactly zero in both",
    ):
        assert claim in flat, f"the caption no longer says {claim!r}"
