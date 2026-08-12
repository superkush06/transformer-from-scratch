"""The animated learning figure has to survive being served by GitHub.

``docs/learning_light.svg`` and its dark twin are the first thing on the front
page, fetched from raw.githubusercontent.com, which serves them under
``default-src 'none'; style-src 'unsafe-inline'; sandbox``. Under that policy
SMIL runs, JavaScript does not, and nothing external loads: no fonts, no
images, not even a data: URI. A figure that breaks one of those rules does not
fail loudly. It renders as a still frame, or as a blank box, on the first page
anyone sees.

So this module checks the properties the renderer cares about rather than the
picture. It reads the committed files and never runs the generator, because
generating them trains a model and CI should not pay for that twice.

The last two tests are about the caption instead. The README quotes the loop
timing and the first and last frames off these files, and a reader who arrives
mid-loop reads the caption before they read the picture, so a caption
describing text the file does not contain is worse than no caption.

Regenerate with ``PYTHONPATH=. python examples/make_learning_anim.py``.
"""

from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
README = ROOT / "README.md"
FIGURES = ("learning_light.svg", "learning_dark.svg")
SVGNS = "{http://www.w3.org/2000/svg}"

# Platane's contribution snake is 97 KB and is the largest animated SVG in
# common use on a profile page. Half of that is a comfortable ceiling.
SIZE_CAP = 60 * 1024

REGEN = "regenerate with `PYTHONPATH=. python examples/make_learning_anim.py`"


@pytest.fixture(params=FIGURES)
def svg(request) -> str:
    return (DOCS / request.param).read_text()


def test_parses_as_xml(svg):
    ET.fromstring(svg)


def test_fits_in_the_size_budget(svg):
    assert len(svg.encode()) < SIZE_CAP, REGEN


def test_actually_animates(svg):
    root = ET.fromstring(svg)
    tags = {e.tag.split("}")[-1] for e in root.iter()}
    assert tags & {"animate", "animateTransform", "animateMotion"}, REGEN


def test_carries_no_script(svg):
    assert "<script" not in svg.lower()
    assert not re.search(r"\son\w+\s*=", svg), "no inline event handlers"


def test_fetches_nothing_from_outside(svg):
    """Every url() and href must be a same-document fragment."""
    for ref in re.findall(r"url\(([^)]*)\)", svg) + re.findall(r'href="([^"]*)"', svg):
        assert ref.startswith("#"), f"external reference {ref!r}"
    assert "data:" not in svg
    assert "@font-face" not in svg


def test_names_a_font_stack_not_a_font(svg):
    """GitHub blocks webfonts, so every family has to end in a generic."""
    families = re.findall(r'font-family="([^"]*)"', svg)
    assert families
    for fam in families:
        last = fam.rsplit(",", 1)[-1].strip().strip("'\"")
        assert last in {"serif", "sans-serif", "monospace"}, fam


def test_the_terminal_text_keeps_its_whitespace(svg):
    """The frames are hard wrapped to a fixed column, and the early ones are
    mostly runs of spaces. Without xml:space the renderer collapses those and
    the character grid the cross-fade depends on comes apart."""
    assert ET.fromstring(svg).get("{http://www.w3.org/XML/1998/namespace}space") \
        == "preserve", REGEN


def test_exactly_one_frame_is_opaque_before_the_animation_starts(svg):
    """A renderer that ignores SMIL shows a still, and the still should be the
    finished text rather than twelve frames stacked on top of each other."""
    frames = [g for g in ET.fromstring(svg).iter(f"{SVGNS}g")
              if g.get("opacity") is not None]
    assert len(frames) >= 10, REGEN
    assert [g.get("opacity") for g in frames].count("1") == 1, REGEN


def test_the_frames_cover_the_loop_without_gaps(svg):
    """Every frame's `animate` shares one duration, and their visible windows
    tile the loop: the sum of the opacities is 1 at every keyTime, so there is
    never a moment with no text and never a moment with two texts at full
    strength."""
    frames = [g for g in ET.fromstring(svg).iter(f"{SVGNS}g")
              if g.get("opacity") is not None]
    tracks = []
    for g in frames:
        a = g.find(f"{SVGNS}animate")
        assert a is not None and a.get("attributeName") == "opacity", REGEN
        assert a.get("repeatCount") == "indefinite", REGEN
        tracks.append((
            a.get("dur"),
            [float(v) for v in a.get("values").split(";")],
            [float(k) for k in a.get("keyTimes").split(";")],
        ))
    assert len({dur for dur, _, _ in tracks}) == 1, REGEN

    def opacity(values, times, t):
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                span = times[i + 1] - times[i]
                if span == 0:
                    continue
                f = (t - times[i]) / span
                return values[i] + f * (values[i + 1] - values[i])
        return values[-1]

    for t in sorted({k for _, _, ks in tracks for k in ks}):
        total = sum(opacity(v, k, t) for _, v, k in tracks)
        assert abs(total - 1.0) < 1e-9, f"opacity sums to {total} at t={t}"


# --------------------------------------------------------------------------
# the caption has to describe the file
# --------------------------------------------------------------------------

def _readme_flat() -> str:
    return re.sub(r"\s+", " ", README.read_text())


def _frame_text(svg: str, which: int) -> str:
    """Reassemble one frame's wrapped lines back into a single string."""
    frames = [g for g in ET.fromstring(svg).iter(f"{SVGNS}g")
              if g.get("opacity") is not None]
    lines = [t for t in frames[which].iter(f"{SVGNS}text")
             if t.get("font-size") == "16"]
    return "".join("".join(t.itertext()) for t in lines)


def test_the_readme_quotes_the_frames_the_figure_holds(svg):
    """The README shows what step 0 wrote and what the last step wrote. Both
    have to appear in the frames actually in the file, or the prose is
    describing a run that no longer exists.

    Read against the raw README rather than the flattened one, so a quotation
    has to sit on one source line: the samples contain runs of two and three
    spaces, and flattening a wrapped line would silently make a mangled
    quotation match.
    """
    quoted = set(re.findall(r"`([^`]{40,})`", README.read_text()))
    assert quoted, "the README no longer quotes any frame"
    for which, what in ((0, "first"), (-1, "last")):
        drawn = _frame_text(svg, which)
        assert any(q in drawn for q in quoted), \
            f"the README quotes nothing from the {what} frame; {REGEN}"


def test_the_readme_quotes_the_real_loop_and_hold(svg):
    """The seconds in the caption come off the SMIL clock, not off a guess."""
    frames = [g for g in ET.fromstring(svg).iter(f"{SVGNS}g")
              if g.get("opacity") is not None]
    animates = [g.find(f"{SVGNS}animate") for g in frames]
    dur = float(animates[0].get("dur").rstrip("s"))
    # the last frame's plateau: from where it reaches 1 to where it leaves it
    values = [float(v) for v in animates[-1].get("values").split(";")]
    times = [float(k) for k in animates[-1].get("keyTimes").split(";")]
    up = [times[i] for i, v in enumerate(values) if v == 1.0]
    hold = (up[-1] - up[0]) * dur

    said = re.search(r"loop is (\d+\.\d+) seconds.{0,90}?holds the last "
                     r"frame for (\d+\.\d+)", _readme_flat())
    assert said, "the README no longer states the loop and the hold"
    assert float(said.group(1)) == round(dur, 1)
    assert float(said.group(2)) == round(hold, 1)
