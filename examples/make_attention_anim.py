"""Draw every attention head sharpening as the character model trains.

    PYTHONPATH=. python3 examples/make_attention_anim.py

Writes docs/attention_light.svg and docs/attention_dark.svg: one heatmap per
head, all of them animating together across five checkpoints of one training
run.

The weights are not invented and not rearranged. This script calls the same
driver the browser demo calls, `docs/demo/tfsdemo.py`, trains the character
model on the text that demo starts with, and reads each head's `(T, T)`
attention matrix straight out of the forward cache at each checkpoint. The
loss printed alongside comes from the demo's own next-character distribution,
so every number in the figure is one the page would show you.

Two things are worth watching, and they disagree with each other:

* the upper right triangle of every grid stays empty, because position i may
  only read positions up to i. That is the causal mask, and it is a property
  of the code rather than of the training;
* the lower triangle starts nearly flat, because a freshly initialised head
  spreads its weight almost evenly over everything the mask allows, and then
  concentrates. The bar under each grid measures how far that goes: it is the
  mean over rows of exp(row entropy), the number of positions the row is
  effectively reading. A uniform causal row of length T averages (T+1)/2.

Size is the constraint. GitHub serves these under a CSP that kills
JavaScript, external fonts and data URIs, and a README should not pull a
megabyte, so the budget is 60 KB per file. Three things buy that: opacity is
quantised to the five levels the legend prints, plus invisible; cells that
share a quantised trajectory share one CSS animation; and cells whose
trajectory never changes are drawn once with no animation at all. Everything
is generated from the run, so re-running with the same seeds rewrites the same
bytes.
"""

from __future__ import annotations

import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DEMO = DOCS / "demo"
sys.path.insert(0, str(DEMO))

import tfsdemo  # noqa: E402

# ---------------------------------------------------------------------------
# what to run
# ---------------------------------------------------------------------------

# The demo's own architecture and text, so the figure is a picture of the
# thing a reader can drive in their browser rather than a private setup.
ARCH = dict(d_model=48, n_heads=4, n_blocks=2, d_ff=96, ctx=32, batch=16,
            lr=3e-3, seed=0)

# Log-spaced, because the interesting movement is early and the last stretch
# only settles it. Five of them: enough to read as motion, few enough that the
# file fits.
CHECKPOINTS = (0, 50, 200, 700, 3000)

# How many characters of the corpus to attend over. The grid is T^2 cells per
# head, so this is the knob that decides whether the file fits: 12 characters
# give 78 drawable cells per head once the mask has taken half, 16 give 136.
WINDOW = 12

# Opacity levels above zero. Six values total including "invisible", and the
# legend prints the weight each one stands for.
LEVELS = 5

# opacity = w ** GAMMA. A trained row puts most of its mass on one or two
# positions and an untrained one holds about 1/T everywhere, and a linear ramp
# renders 1/12 as almost nothing, which would make the untrained head look
# empty rather than flat. The legend prints the weight each level stands
# for so the compression is on the page, not hidden in this file.
GAMMA = 0.6

# ---------------------------------------------------------------------------
# geometry, in user units
# ---------------------------------------------------------------------------

PITCH, CELL = 13, 12          # cell pitch and cell size, so a 1-unit gap
COLS = 4                      # heads across; one row of grids per block
MARGIN = 26
ROWLAB = 14                   # gutter for the row characters
GAP_X, GAP_Y = 28, 30
TOP = 182                     # title, subtitle, clock
HEAD_ABOVE = 30               # head label and the column characters
HEAD_BELOW = 36               # the spread readout and its bar

DUR = 13                      # seconds per loop
HOLD, STEP = 9, 14            # percent: each checkpoint holds, then morphs

FONT = "ui-monospace,SFMono-Regular,Menlo,monospace"

LIGHT = dict(bg="#F7F4EF", ink="#35322C", quiet="#8B857A", hair="#DAD4C9",
             blue="#1B6CA8", rust="#C05F1B")
DARK = dict(bg="#0D1117", ink="#C9D1D9", quiet="#8B949E", hair="#30363D",
            blue="#58A6FF", rust="#DB8B4F")


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def probe_positions(text: str, ctx: int, stride: int = 7) -> list[tuple[str, str]]:
    """(prefix, next character) pairs to score the model on, fixed for the run."""
    return [(text[max(0, i - ctx):i], text[i])
            for i in range(ctx, len(text), stride)]


def probe_loss(probes: list[tuple[str, str]]) -> float:
    """Mean -log p(next character), in nats, through the demo's own API.

    `gpt_next` is what the browser page draws its next-character bars from, so
    this is the same distribution a reader sees. It is not held out: `gpt_step`
    draws its windows from anywhere in the corpus, so the model has been shown
    every one of these positions, and the figure calls the number "loss on its
    own text" rather than anything stronger.
    """
    total = 0.0
    for prefix, want in probes:
        dist = tfsdemo.gpt_next(prefix, k=len(tfsdemo.CORPUS["itos"]))
        p = next((row["p"] for row in dist["top"] if row["ch"] == want), 0.0)
        total += -math.log(max(p, 1e-12))
    return total / len(probes)


def spread(w: np.ndarray) -> float:
    """Mean over rows of exp(row entropy): positions the row effectively reads.

    One number for "how concentrated is this head", on the scale of the thing
    being counted. A row that splits evenly over its k legal positions scores
    k; a row that puts everything on one position scores 1.
    """
    out = []
    for i in range(w.shape[0]):
        row = w[i, : i + 1]
        p = row[row > 1e-12]
        out.append(float(np.exp(-(p * np.log(p)).sum())))
    return float(np.mean(out))


def capture() -> dict:
    """Train once, and keep every head's matrix at each checkpoint."""
    corpus = tfsdemo.corpus_set()
    arch = tfsdemo.gpt_begin(**ARCH)
    text = tfsdemo.CORPUS["text"]
    window = text[:WINDOW]
    probes = probe_positions(text, arch["ctx"])

    frames, at = [], 0
    for target in CHECKPOINTS:
        if target > at:
            tfsdemo.gpt_step(target - at)
            at = target
        shot = tfsdemo.gpt_attention(window)
        heads = [{"block": h["block"], "head": h["head"],
                  "w": np.array(h["w"], dtype=float)} for h in shot["heads"]]
        for h in heads:
            h["spread"] = spread(h["w"])
        # Greedy, so this is the model's own best guess rather than a lucky
        # sample: the least deniable evidence that the run went anywhere.
        wrote = tfsdemo.gpt_sample(prompt=window, n=WINDOW, temperature=0.0)
        frames.append({"step": at, "loss": probe_loss(probes), "heads": heads,
                       "wrote": wrote["text"][len(window):]})
        print(f"  step {at:5d}  loss {frames[-1]['loss']:.3f} nats  "
              f"writes {frames[-1]['wrote']!r}  spread "
              + " ".join(f"{h['spread']:5.2f}" for h in heads))

    chars = shot["chars"]
    return {"chars": chars, "window": window, "frames": frames, "arch": arch,
            "corpus": corpus, "n_blocks": ARCH["n_blocks"],
            "n_heads": ARCH["n_heads"], "uniform": (len(chars) + 1) / 2.0}


# ---------------------------------------------------------------------------
# quantising, and the animation that falls out of it
# ---------------------------------------------------------------------------

def level(w: float) -> int:
    """Attention weight to opacity level, 0 (invisible) to LEVELS."""
    return int(round(min(1.0, max(0.0, w)) ** GAMMA * LEVELS))


def level_weight(k: int) -> float:
    """The weight a level stands for, for the legend to print."""
    return (k / LEVELS) ** (1.0 / GAMMA)


def weight_label(k: int) -> str:
    """That weight, spelled for a 4-character legend slot."""
    w = level_weight(k)
    return "1" if k >= LEVELS else f"{w:.2f}".lstrip("0")


def opacity(k: int) -> str:
    """Shortest CSS spelling of a level's opacity.

    Level 0 has to spell itself "0" and not "." Trimming both ends of "0.00"
    leaves a bare dot, which is not a CSS number, so the browser throws the
    declaration away and the keyframe it sat in ends up empty. A keyframe with
    no opacity in it is not a keyframe that means "invisible": if the empty one
    is the last stop, the animation has no 100% for opacity, falls back to the
    element's own value of 1, and a cell whose weight went to nothing ramps up
    to the brightest thing in the grid over the hold. That is the figure saying
    the opposite of the run.
    """
    if k <= 0:
        return "0"
    if k >= LEVELS:
        return "1"
    return f"{k / LEVELS:.2f}".rstrip("0").lstrip("0")


def phase_stops(n: int) -> list[tuple[int, int]]:
    """Percent windows each checkpoint holds for, before morphing to the next.

    The last one runs to 100, so the loop ends on a still: with five
    checkpoints the trained heads hold from 56% to the end, which is 5.7 of the
    13 seconds, and that is what whoever arrives mid-cycle most likely sees.
    """
    return [(k * STEP, k * STEP + HOLD) if k < n - 1 else (k * STEP, 100)
            for k in range(n)]


def keyframes(name: str, traj: tuple[int, ...], stops) -> str:
    """One CSS animation for one quantised trajectory.

    Consecutive checkpoints at the same level are merged into a single hold,
    which is most of why this fits: a cell that stays faint for four
    checkpoints costs one stop, not four.
    """
    parts, k = [], 0
    while k < len(traj):
        j = k
        while j + 1 < len(traj) and traj[j + 1] == traj[k]:
            j += 1
        parts.append(f"{stops[k][0]}%,{stops[j][1]}%{{opacity:{opacity(traj[k])}}}")
        k = j + 1
    return f"@keyframes {name}{{{''.join(parts)}}}"


def name_of(i: int) -> str:
    """Short animation names for the trajectories: A, B, ... Z, A1, B1, ...

    Upper case and always letter-then-digits, which is what keeps them out of
    the way of the hand-written class names below (`fq`, `ph2`). Getting that
    wrong is not a cosmetic bug: a trajectory that lands on the name of a
    label's animation drives the label instead, and the figure quietly shows
    two checkpoints at once. It did, until this changed.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[i % 26] + ("" if i < 26 else str(i // 26))


def cells_path(cells: list[tuple[int, int]]) -> str:
    """Path data for a set of (row, column) cells that animate together.

    A cell is a horizontal segment of length CELL stroked CELL wide, which
    paints exactly the same square a rectangle would for less than half the
    characters: `m13 0h12` against `m13 0h12v12h-12z`. Moves are relative to
    wherever the last segment left the pen, which is its right-hand end.
    """
    out, px, py = [], 0, 0
    for n, (i, j) in enumerate(sorted(cells)):
        x, y = j * PITCH, i * PITCH + CELL // 2
        out.append(f"M{x} {y}" if n == 0 else f"m{x - px} {y - py}")
        out.append(f"h{CELL}")
        px, py = x + CELL, y
    return "".join(out)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def esc(s: str) -> str:
    """XML-escape a fragment of the corpus before it goes in a label."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def num(v: float) -> str:
    """Trim a coordinate to something short but exact enough to draw."""
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return s or "0"


def keytimes(stops) -> str:
    """SMIL keyTimes for the same hold-then-morph clock the CSS uses."""
    out = []
    for a, b in stops:
        out += [a / 100.0, b / 100.0]
    return ";".join(num(v) if v in (0.0, 1.0) else f"{v:.2f}".lstrip("0")
                    for v in out)


_ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|\w+);")


def advance(text: str, size: float) -> float:
    """How wide a run of monospace is, near enough to catch an overflow.

    Every font in the stack advances 0.6 em or a shade over, so 0.62 is a
    ceiling rather than a guess. Entities count as the one character they
    draw. This exists because a sentence that runs off the viewBox is
    invisible in the source and obvious on the page, and the only reliable
    place to catch it is here.
    """
    return 0.62 * size * len(_ENTITY.sub("x", text))


def render(cap: dict, pal: dict) -> str:
    T = len(cap["chars"])
    frames = cap["frames"]
    stops = phase_stops(len(frames))
    side = T * PITCH
    head_w = ROWLAB + side
    width = 2 * MARGIN + COLS * head_w + (COLS - 1) * GAP_X
    row_h = HEAD_ABOVE + side + HEAD_BELOW
    rows = math.ceil(len(frames[0]["heads"]) / COLS)
    legend_y = TOP + rows * row_h + (rows - 1) * GAP_Y + 34
    height = legend_y + 58

    # --- group cells by quantised trajectory ------------------------------
    # A trajectory is one cell's level at each of the checkpoints. Cells
    # sharing one get one CSS animation for the whole figure; cells that never
    # change get no animation and no class.
    animated: dict[tuple[int, ...], str] = {}
    per_head: list[dict] = []
    for hi in range(len(frames[0]["heads"])):
        moving: dict[tuple[int, ...], list[tuple[int, int]]] = {}
        static: dict[int, list[tuple[int, int]]] = {}
        for i in range(T):
            for j in range(i + 1):
                traj = tuple(level(f["heads"][hi]["w"][i, j]) for f in frames)
                if len(set(traj)) == 1:
                    if traj[0]:
                        static.setdefault(traj[0], []).append((i, j))
                else:
                    moving.setdefault(traj, []).append((i, j))
                    if traj not in animated:
                        animated[traj] = name_of(len(animated))
        per_head.append({"moving": moving, "static": static})

    # Class names are a namespace, and it has to stay disjoint: `fq` and `ph2`
    # are hand-written, `name_of` only ever emits upper case, so a trajectory
    # can never take over a label's animation.
    css = [
        f"text{{fill:{pal['ink']}}}",
        f"path,rect,circle,text{{animation-duration:{DUR}s;"
        "animation-timing-function:linear;animation-iteration-count:infinite}",
        f".fq{{fill:{pal['quiet']}}}",
        f".fr{{fill:{pal['rust']}}}",
        f".fi{{fill:{pal['ink']}}}",
    ]
    for k in range(len(frames)):
        a, b = stops[k]
        if k == 0:
            body = f"0%,{b}%{{opacity:1}}{b + 2}%,100%{{opacity:0}}"
        elif k == len(frames) - 1:
            body = f"0%,{a - 2}%{{opacity:0}}{a}%,100%{{opacity:1}}"
        else:
            body = (f"0%,{a - 2}%{{opacity:0}}{a}%,{b}%{{opacity:1}}"
                    f"{b + 2}%,100%{{opacity:0}}")
        css.append(f".ph{k}{{animation-name:ph{k}}}")
        css.append(f"@keyframes ph{k}{{{body}}}")
    for traj, name in animated.items():
        css.append(f".{name}{{animation-name:{name}}}")
        css.append(keyframes(name, traj, stops))

    out: list[str] = [
        # The font stack rides on the root as an attribute rather than in the
        # stylesheet, so it is inherited even where the CSS is not honoured.
        # GitHub will not load a web font, so the stack has to name what a
        # reader already has.
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="{FONT}" role="img" '
        f'aria-label="Every attention head of a character-level GPT, drawn at '
        f'{len(frames)} checkpoints of one training run">',
        f"<style>{''.join(css)}</style>",
        f'<rect width="{width}" height="{height}" fill="{pal["bg"]}"/>',
    ]

    def put(x, y, size, body, cls="fq", anchor="", origin=0):
        """A run of text, refusing to run off the page.

        `origin` is whatever the enclosing group is translated by, since x is
        local to it and the page edge is not.
        """
        end = origin + x + advance(body, size)
        if anchor == "" and end > width - MARGIN + 8:
            raise ValueError(f"label overruns the {width}-unit page by "
                             f"{end - (width - MARGIN + 8):.0f} units: {body!r}")
        at = f' text-anchor="{anchor}"' if anchor else ""
        klass = f' class="{cls}"' if cls else ""
        out.append(f'<text x="{num(x)}" y="{num(y)}" font-size="{size}"'
                   f'{klass}{at}>{body}</text>')

    # --- title and subtitle ------------------------------------------------
    arch, corpus = cap["arch"], cap["corpus"]
    put(MARGIN, 34, 19, "Attention sharpening, one training run", "")
    for k, line in enumerate([
        f"Every head of a {cap['n_blocks']}-block, {cap['n_heads']}-head "
        f"character GPT, {arch['params']:,} parameters, every backward pass "
        "by hand.",
        f"Trained on {corpus['chars']} characters. Cell opacity is the "
        f"attention weight over the {T} below, from the forward cache.",
        "The upper triangle stays empty throughout, because position i may "
        "only read positions up to i.",
    ]):
        put(MARGIN, 58 + 16 * k, 11.5, line)

    # --- the clock ---------------------------------------------------------
    rule_y = 130
    x0, x1 = MARGIN + 76, width - MARGIN - 8
    ticks = [x0 + (x1 - x0) * k / (len(frames) - 1) for k in range(len(frames))]
    put(MARGIN, rule_y + 4, 10.5, "steps")
    out.append(f'<path d="M{num(x0)} {rule_y}H{num(x1)}" stroke="{pal["hair"]}" '
               'stroke-width="1" fill="none"/>')
    for k, (f, x) in enumerate(zip(frames, ticks, strict=True)):
        out.append(f'<path d="M{num(x)} {rule_y - 4}v8" stroke="{pal["hair"]}" '
                   'stroke-width="1"/>')
        put(x, rule_y - 10, 10.5, f"{f['step']:,}", "fq", "middle")
        put(x, rule_y - 10, 10.5, f"{f['step']:,}", f"fr ph{k}", "middle")
        put(x0, rule_y + 21, 10.5,
            f"loss on its own text {f['loss']:.2f} nats, and prompted "
            f"&#8220;{esc(cap['window'])}&#8221; it writes "
            f"&#8220;{esc(f['wrote'])}&#8221;", f"ph{k}")
    kt = keytimes(stops)
    cx = ";".join(num(x) for x in ticks for _ in (0, 1))
    out.append(f'<circle cy="{rule_y}" r="3.2" fill="{pal["rust"]}">'
               f'<animate attributeName="cx" values="{cx}" keyTimes="{kt}" '
               f'dur="{DUR}s" repeatCount="indefinite"/></circle>')

    # --- one grid per head -------------------------------------------------
    # Both axes are one <text> with a position per character: an x list for the
    # keys along the top, and x *and* y lists down the side, because a y list
    # alone would leave every character advancing to the right of the last.
    labels = "".join(cap["chars"])
    col_x = " ".join(num(j * PITCH + CELL / 2) for j in range(T))
    row_x = " ".join(["-6"] * T)
    row_y = " ".join(str(i * PITCH + CELL - 3) for i in range(T))
    for hi, head in enumerate(frames[0]["heads"]):
        gx = MARGIN + ROWLAB + (hi % COLS) * (head_w + GAP_X)
        gy = TOP + HEAD_ABOVE + (hi // COLS) * (row_h + GAP_Y)
        out.append(f'<g transform="translate({num(gx)} {num(gy)})">')
        out.append(f'<text x="0" y="-22" font-size="11.5" class="fi">block '
                   f'{head["block"]} &#183; head {head["head"]}</text>')
        out.append(f'<text x="{col_x}" y="-7" font-size="9" '
                   f'text-anchor="middle" class="fq">{labels}</text>')
        out.append(f'<text x="{row_x}" y="{row_y}" font-size="9" '
                   f'text-anchor="end" class="fq">{labels}</text>')

        out.append(f'<g fill="none" stroke-width="{CELL}" '
                   f'stroke="{pal["rust"]}">')
        # the mask: what this head is not allowed to read, one band per row
        strips = []
        for i in range(T - 1):
            x = (i + 1) * PITCH
            strips.append(f"M{x} {i * PITCH + CELL // 2}"
                          f"h{(T - i - 2) * PITCH + CELL}")
        out.append(f'<path stroke="{pal["hair"]}" d="{"".join(strips)}"/>')

        for lv, cells in sorted(per_head[hi]["static"].items()):
            op = "" if lv >= LEVELS else f' opacity="{opacity(lv)}"'
            out.append(f'<path{op} d="{cells_path(cells)}"/>')
        for traj, cells in per_head[hi]["moving"].items():
            out.append(f'<path class="{animated[traj]}" '
                       f'd="{cells_path(cells)}"/>')
        out.append("</g>")

        # spread: the measured number, and a bar so it reads without being read
        bar_w = side - 2
        vals, mark = [], bar_w * min(1.0, cap["uniform"] / T)
        for f in frames:
            v = f["heads"][hi]["spread"] / T * bar_w
            vals += [num(v), num(v)]
        out.append(f'<path d="M0 {side + 26}h{bar_w}" stroke="{pal["hair"]}" '
                   'stroke-width="5" fill="none"/>')
        out.append(f'<rect x="0" y="{side + 23.5}" height="5" width="0" '
                   f'fill="{pal["rust"]}"><animate attributeName="width" '
                   f'values="{";".join(vals)}" keyTimes="{kt}" dur="{DUR}s" '
                   'repeatCount="indefinite"/></rect>')
        out.append(f'<path d="M{num(mark)} {side + 21}v9" '
                   f'stroke="{pal["blue"]}" stroke-width="1.2" fill="none"/>')
        for k, f in enumerate(frames):
            put(0, side + 16, 10.5,
                f"reads {f['heads'][hi]['spread']:.1f} positions",
                f"fi ph{k}", origin=gx)
        out.append("</g>")

    # --- legend ------------------------------------------------------------
    lx = MARGIN
    put(lx, legend_y - 14, 10.5, "attention weight")
    for k in range(1, LEVELS + 1):
        x = lx + (k - 1) * 48
        out.append(f'<rect x="{x}" y="{legend_y - 8}" width="18" height="12" '
                   f'fill="{pal["rust"]}" opacity="{opacity(k)}"/>')
        put(x + 22, legend_y + 2, 9.5, weight_label(k))
    x = lx + LEVELS * 48 + 8
    out.append(f'<rect x="{x}" y="{legend_y - 8}" width="18" height="12" '
               f'fill="{pal["hair"]}"/>')
    put(x + 22, legend_y + 2, 9.5, "masked")
    put(lx, legend_y + 26, 10.5,
        f"Bar under each grid: how many of the {T} positions the head "
        f"effectively reads, the mean over rows of exp(row entropy).")
    put(lx, legend_y + 42, 10.5,
        f"The mark on it is {cap['uniform']:.1f}, what a head spread evenly "
        f"over everything the mask allows would score.")
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------

def main() -> None:
    print(f"training {ARCH['n_blocks']}x{ARCH['n_heads']} heads, "
          f"checkpoints at {', '.join(str(c) for c in CHECKPOINTS)}:")
    cap = capture()
    for theme, pal in (("light", LIGHT), ("dark", DARK)):
        path = DOCS / f"attention_{theme}.svg"
        svg = render(cap, pal)
        path.write_text(svg)
        print(f"wrote {path.relative_to(ROOT)}  {len(svg.encode()):,} bytes")
    first, last = cap["frames"][0], cap["frames"][-1]
    for a, b in zip(first["heads"], last["heads"], strict=True):
        print(f"  block {a['block']} head {a['head']}: reads "
              f"{a['spread']:.2f} positions at step 0, {b['spread']:.2f} at "
              f"{last['step']:,} (uniform {cap['uniform']:.1f})")


if __name__ == "__main__":
    main()
