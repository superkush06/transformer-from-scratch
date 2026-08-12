"""Draw one training run as an animated SVG: text learning to write.

    PYTHONPATH=. python examples/make_learning_anim.py

Writes docs/learning_light.svg and docs/learning_dark.svg.

A character-level GPT on 353 characters of Hamlet crosses from noise into
English in a few hundred steps, and that crossing is the most interesting
thing in this repository to actually watch. So this figure is one run,
sampled twelve times: 320 characters of generated text per frame, hard
wrapped to 64 columns the way a terminal wraps, cross-faded frame to frame
so the characters morph in place. Beneath the text is the loss curve of the
same run with a playhead on the sampled step, because the text on its own
says nothing about what it cost.

Every frame comes out of `docs/demo/tfsdemo.py`, the same driver the browser
demo calls: `corpus_set`, `gpt_begin`, `gpt_step`, `gpt_sample`. Nothing here
is retouched. The first frame is the model before a single update, and it is
unreadable, which is the point of showing it.

`losses[i]` is the cross-entropy under the weights after `i` updates, so a
frame's step and its loss describe the same weights that produced that
frame's text. The generator takes one step past the last frame so that the
last frame's loss is a measured number rather than an extrapolated one.

Constraints this file is written against: GitHub serves an SVG from
raw.githubusercontent.com under `default-src 'none'; style-src
'unsafe-inline'; sandbox`. SMIL animates, JavaScript does not run, and
nothing external loads: no fonts, no images, not even a data: URI. So
everything is inline and the font is a stack rather than a font. Both files
stay well under 60 KB and the README pairs them with `<picture>`, because
`prefers-color-scheme` inside an SVG loaded through `<img>` follows the
operating system rather than GitHub's own theme toggle.
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
sys.path.insert(0, str(DOCS / "demo"))

import tfsdemo as demo  # noqa: E402

# ---- the run ---------------------------------------------------------------
SEED = 0
TEMPERATURE = 0.4
PROMPT = "to be"
COLS, ROWS = 64, 5

# Passed straight to `gpt_begin`, so the numbers in the subtitle are the
# numbers the model was built with rather than a caption that can drift.
ARCH = dict(d_model=48, n_heads=4, n_blocks=2, d_ff=96, ctx=32, batch=16,
            lr=3e-3, seed=SEED)

# Twelve steps, bunched where the run is actually changing. Between 15 and
# 200 the model goes from stuttering one letter at a time to reciting whole
# clauses, and spacing the frames evenly would spend most of the loop on the
# half of the run where nothing new happens.
FRAMES = (0, 15, 30, 50, 75, 100, 130, 170, 220, 300, 420, 600)

# ---- animation -------------------------------------------------------------
# The cross-fade is what makes this readable. Every frame occupies the same
# character grid, so an outgoing character dissolves into the incoming one in
# place, and the eye reads the change as one letter becoming another rather
# than as the whole block blinking. Keep CF short: while it runs, the step and
# loss readouts are two numbers on top of each other.
DWELL = 0.95       # seconds each frame is up
HOLD = 2.2         # extra seconds on the last frame, so it reads as a still
CF = 0.25          # cross-fade
# Rounded, because 12 * 0.95 + 2.2 is 13.599999999999998 in binary floating
# point and that spelling would go into the file twelve times.
DUR = round(len(FRAMES) * DWELL + HOLD, 3)

# ---- geometry, user units --------------------------------------------------
# EM is the widest advance-per-em in the stack below: SF Mono is 0.600,
# Menlo and DejaVu Sans Mono 0.6022, Consolas 0.5498. Sizing to the widest
# means a narrower fallback leaves slack rather than overflowing the panel,
# and `wide()` asserts every run of text still fits.
EM = 0.6025
FS = 16            # the terminal type size
CW = EM * FS       # advance width of one character
LH = 30            # line height; generous on purpose, this is meant to be read
M = 34             # page margin
PAD = 22           # panel padding
PW = round(COLS * CW) + 2 * PAD
W = 2 * M + PW

PT = 100           # top of the text panel
SB = 34            # height of its status strip
TT = PT + SB + 28  # first text baseline
PB = TT + (ROWS - 1) * LH + 18      # bottom of the text panel

QT = PB + 16       # top of the loss panel
PLX = M + PAD + 26                  # loss plot, left
PRX = M + PW - PAD                  # loss plot, right
LY0 = QT + 20                       # loss plot, top
LOSSH = 74
LY1 = LY0 + LOSSH
QB = LY1 + 40      # bottom of the loss panel
H = QB + 80        # a three-line caption under it

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

THEMES = {
    "light": dict(page="#F7F4EF", ink="#35322C", quiet="#8B857A",
                  hair="#DAD4C9", blue="#1B6CA8", rust="#C05F1B"),
    "dark": dict(page="#0D1117", ink="#C9D1D9", quiet="#8B949E",
                 hair="#30363D", blue="#58A6FF", rust="#DB8B4F"),
}


# ---- capture ---------------------------------------------------------------

class Run:
    """One training run, reduced to what the picture needs."""

    def __init__(self) -> None:
        corpus = demo.corpus_set()
        arch = demo.gpt_begin(**ARCH)

        losses: list[float] = []
        samples: list[str] = []
        at = 0
        for step in FRAMES:
            if step > at:
                losses.extend(demo.gpt_step(step - at)["losses"])
                at = step
            got = demo.gpt_sample(prompt=PROMPT, n=COLS * ROWS - len(PROMPT),
                                  temperature=TEMPERATURE)
            assert got["step"] == step and not got["dropped"]
            samples.append(got["text"])
        # `losses[i]` is measured under the weights after i updates, so the
        # last frame needs one more step taken before its own loss exists.
        losses.extend(demo.gpt_step(1)["losses"])

        self.corpus = corpus
        self.arch = arch
        self.losses = losses
        self.samples = samples
        assert len(losses) == FRAMES[-1] + 1
        assert all(len(s) == COLS * ROWS for s in samples)

    def loss_at(self, step: int) -> float:
        return self.losses[step]

    @property
    def uniform(self) -> float:
        """What guessing uniformly from the alphabet costs, in nats."""
        return math.log(self.corpus["vocab"])


# ---- formatting ------------------------------------------------------------

def n(v: float) -> str:
    """Shortest faithful spelling of a coordinate."""
    r = round(v, 1)
    return str(int(r)) if r == int(r) else f"{r:.1f}"


def frac(t: float) -> str:
    """A keyTimes entry: a fraction of the loop, trimmed."""
    if t <= 0:
        return "0"
    if t >= 1:
        return "1"
    return f"{t:.4f}".rstrip("0").rstrip(".")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wide(s: str, size: float) -> float:
    """How wide a run of monospace text is, at the stack's widest advance.

    Every full-width line goes through this, because an overlong title does
    not wrap in an SVG: it runs off the right edge and out of the viewBox,
    and it does it silently.
    """
    return len(s) * EM * size


# ---- the cross-fade schedule ----------------------------------------------

def fade(i: int, last: int) -> str:
    """One `animate` driving frame i's opacity over the whole loop.

    Every frame holds for DWELL and then hands over in CF seconds, so the
    outgoing and incoming frames are always mid-fade together and the
    characters appear to morph in place rather than blink. The last frame
    holds for HOLD longer and then hands back to the first, which is what
    makes the loop's seam land on the finished text.
    """
    start = i * DWELL
    end = DUR if i == last else start + DWELL
    if i == 0:
        values = "1;1;0;0;1"
        times = [0.0, (DWELL - CF) / DUR, DWELL / DUR, (DUR - CF) / DUR, 1.0]
    elif i == last:
        values = "0;0;1;1;0"
        times = [0.0, (start - CF) / DUR, start / DUR, (DUR - CF) / DUR, 1.0]
    else:
        values = "0;0;1;1;0;0"
        times = [0.0, (start - CF) / DUR, start / DUR,
                 (end - CF) / DUR, end / DUR, 1.0]
    return (f'<animate attributeName="opacity" values="{values}" '
            f'keyTimes="{";".join(frac(t) for t in times)}" '
            f'dur="{DUR}s" repeatCount="indefinite"/>')


# ---- the loss curve --------------------------------------------------------

def loss_path(run: Run, ymax: float) -> str:
    """Every step of the run, one polyline. Absolute points with an implicit
    lineto, which is the cheapest spelling of 601 samples."""
    sx = (PRX - PLX) / FRAMES[-1]
    pts = []
    for step, v in enumerate(run.losses):
        y = LY1 - min(v, ymax) / ymax * LOSSH
        pts.append(f"{n(PLX + step * sx)},{n(y)}")
    return "M" + " ".join(pts)


# ---- the drawing -----------------------------------------------------------

def build(run: Run, theme: str) -> str:
    t = THEMES[theme]
    ymax = math.ceil(max(run.losses) * 2) / 2
    sx = (PRX - PLX) / FRAMES[-1]

    def xstep(step: int) -> float:
        return PLX + step * sx

    def yloss(v: float) -> float:
        return LY1 - min(v, ymax) / ymax * LOSSH

    o: list[str] = []
    o.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="{MONO}" xml:space="preserve">'
    )
    o.append(f'<rect width="{W}" height="{H}" fill="{t["page"]}"/>')

    def full(y: float, size: float, fill: str, s: str) -> None:
        """A line of text across the whole measure, checked for overflow."""
        assert wide(s, size) <= PW, f"{wide(s, size):.0f} > {PW}: {s!r}"
        o.append(f'<text x="{M}" y="{n(y)}" font-size="{size}" '
                 f'fill="{fill}">{esc(s)}</text>')

    # --- title and the facts of the run
    full(40, 18, t["ink"],
         f'Text learning to write: one {FRAMES[-1]}-step run, '
         f'sampled {len(FRAMES)} times')
    full(64, 12.5, t["quiet"],
         f'{run.arch["params"]:,} parameters, {ARCH["d_model"]} wide, '
         f'{ARCH["n_blocks"]} blocks, {ARCH["n_heads"]} heads, '
         f'{run.arch["ctx"]}-char context, batch {run.arch["batch"]}, '
         f'Adam at {ARCH["lr"]:g}')
    full(82, 12.5, t["quiet"],
         f'a {run.corpus["vocab"]}-character vocabulary built from '
         f'{run.corpus["chars"]} characters of Hamlet, seed {SEED}')

    # --- the terminal panel
    o.append(
        f'<rect x="{M}" y="{PT}" width="{PW}" height="{PB - PT:.0f}" '
        f'rx="6" fill="none" stroke="{t["hair"]}"/>'
    )
    # the status strip, with the panel's own top corners rounded into it
    o.append(
        f'<path d="M{M},{PT + 6}a6,6 0 0 1 6,-6h{PW - 12}a6,6 0 0 1 6,6'
        f'v{SB - 6}H{M}z" fill="{t["hair"]}" fill-opacity=".45"/>'
        f'<path d="M{M},{PT + SB}h{PW}" stroke="{t["hair"]}"/>'
    )
    sx0 = M + PAD
    o.append(
        f'<text x="{sx0}" y="{PT + 22}" font-size="13" fill="{t["quiet"]}">step</text>'
        f'<text x="{n(sx0 + 15 * CW)}" y="{PT + 22}" font-size="13" '
        f'fill="{t["quiet"]}">loss</text>'
        f'<text x="{PRX}" y="{PT + 22}" font-size="13" text-anchor="end" '
        f'fill="{t["quiet"]}">{COLS * ROWS} characters at temperature '
        f'{TEMPERATURE}</text>'
    )

    # --- one group per frame: its readout, its text, its playhead
    last = len(FRAMES) - 1
    for i, step in enumerate(FRAMES):
        loss = run.loss_at(step)
        text = run.samples[i]
        g = [f'<g opacity="{1 if i == last else 0}">', fade(i, last)]
        g.append(
            f'<text x="{n(sx0 + 9 * CW)}" y="{PT + 22}" font-size="13" '
            f'text-anchor="end" fill="{t["ink"]}">{step}</text>'
            f'<text x="{n(sx0 + 25 * CW)}" y="{PT + 22}" font-size="13" '
            f'text-anchor="end" fill="{t["rust"]}">{loss:.3f}</text>'
        )
        for r in range(ROWS):
            line = esc(text[r * COLS:(r + 1) * COLS])
            y = TT + r * LH
            if r == 0:
                body = (f'<tspan fill="{t["blue"]}">{line[:len(PROMPT)]}</tspan>'
                        f'<tspan fill="{t["ink"]}">{line[len(PROMPT):]}</tspan>')
            else:
                body = line
            g.append(
                f'<text x="{sx0}" y="{y}" font-size="{FS}" '
                f'fill="{t["ink"]}">{body}</text>'
            )
        # the playhead on the loss curve below: which step wrote this text,
        # and what it was still paying for it
        x, y = xstep(step), yloss(loss)
        g.append(
            f'<path d="M{n(x)},{LY0}V{n(y)}m0,{n(LY1 - y)}v5" '
            f'stroke="{t["rust"]}" stroke-width="1.2" stroke-opacity=".5"/>'
            f'<circle cx="{n(x)}" cy="{n(y)}" r="3.6" fill="{t["rust"]}" '
            f'stroke="{t["page"]}" stroke-width="1.4"/>'
        )
        g.append("</g>")
        o.append("".join(g))

    # --- the loss panel
    o.append(
        f'<rect x="{M}" y="{QT}" width="{PW}" height="{QB - QT:.0f}" '
        f'rx="6" fill="none" stroke="{t["hair"]}"/>'
    )
    for v in range(int(ymax) + 1):
        y = yloss(v)
        o.append(
            f'<path d="M{PLX},{n(y)}H{PRX}" stroke="{t["hair"]}" '
            f'stroke-opacity=".7"/>'
            f'<text x="{PLX - 8}" y="{n(y + 4)}" font-size="11" '
            f'text-anchor="end" fill="{t["quiet"]}">{v}</text>'
        )
    o.append(
        f'<text transform="translate({PLX - 26},{(LY0 + LY1) / 2}) rotate(-90)" '
        f'font-size="11" text-anchor="middle" fill="{t["quiet"]}">nats</text>'
    )
    # what guessing uniformly costs; the run starts above it and has to leave
    yu = yloss(run.uniform)
    o.append(
        f'<path d="M{PLX},{n(yu)}H{PRX}" stroke="{t["quiet"]}" '
        f'stroke-width="1" stroke-dasharray="3 3" stroke-opacity=".8"/>'
        f'<text x="{PRX - 14}" y="{n(yu - 8)}" font-size="11" '
        f'text-anchor="end" fill="{t["quiet"]}">ln {run.corpus["vocab"]} = '
        f'{run.uniform:.3f}, what guessing uniformly costs</text>'
    )
    o.append(
        f'<path d="{loss_path(run, ymax)}" fill="none" stroke="{t["rust"]}" '
        f'stroke-width="1.1" stroke-opacity=".85"/>'
    )
    for step in FRAMES:
        o.append(f'<path d="M{n(xstep(step))},{LY1}v5" stroke="{t["quiet"]}"/>')
    for step in range(0, FRAMES[-1] + 1, 200):
        o.append(
            f'<text x="{n(xstep(step))}" y="{LY1 + 21}" font-size="11" '
            f'text-anchor="middle" fill="{t["quiet"]}">{step}</text>'
        )
    o.append(
        f'<text x="{PRX}" y="{QB - 9}" font-size="11" text-anchor="end" '
        f'fill="{t["quiet"]}">gradient steps; the ticks are the '
        f'{len(FRAMES)} frames above</text>'
    )

    # --- caption
    full(QB + 26, 12.5, t["quiet"],
         f'Every frame is a real sample, hard wrapped at {COLS} columns; '
         f'the prompt is in blue.')
    full(QB + 44, 12.5, t["quiet"],
         "Frame one is the model before any update; loss is cross-entropy "
         "on that step's batch.")
    full(QB + 62, 12.5, t["quiet"],
         'Every gradient behind these updates was derived by hand. There is '
         'no autograd here.')

    o.append("</svg>")
    return "\n".join(o)


def main() -> None:
    run = Run()
    DOCS.mkdir(exist_ok=True)
    for theme in THEMES:
        out = DOCS / f"learning_{theme}.svg"
        svg = build(run, theme)
        out.write_text(svg, encoding="utf-8")
        print(f"wrote {out}  {len(svg.encode()):,} bytes")
    print(f"  {run.arch['params']:,} parameters, vocab {run.corpus['vocab']}, "
          f"{run.corpus['chars']} characters of corpus")
    print(f"  loop {DUR:.1f}s: {len(FRAMES)} frames at {DWELL}s, "
          f"{HOLD}s hold, {CF}s cross-fade")
    print(f"  ln {run.corpus['vocab']} = {run.uniform:.4f} nats")
    for i, step in enumerate(FRAMES):
        print(f"  step {step:>3}  loss {run.loss_at(step):.4f}  "
              f"{run.samples[i][:56]!r}")


if __name__ == "__main__":
    main()
