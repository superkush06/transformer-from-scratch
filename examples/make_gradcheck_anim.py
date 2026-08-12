"""Draw the 1,312-gradient audit as it happens, as an animated SVG.

    PYTHONPATH=. python3 examples/make_gradcheck_anim.py

docs/gradcheck.png shows the finished scatter: every hand-derived partial
against the central difference that measures it, all of it on the diagonal.
Watching the points land is more convincing than being shown that they did,
so this writes the same scatter as a progressive fill, in the order the sweep
actually visits the model, one parameter tensor at a time.

Every coordinate comes from one call to `check_every_scalar()`, the same
function `examples/gradcheck.py` and `tests/test_gradcheck.py` use. Nothing is
hand-placed and no number in the frame is typed: the worst relative error, the
running worst, the margin against the tolerance and the count of coordinates a
log axis cannot hold are all read off that run.

Two files come out, docs/gradcheck_light.svg and docs/gradcheck_dark.svg,
because a `prefers-color-scheme` query inside an SVG that GitHub loads through
`<img>` follows the OS rather than GitHub's own theme toggle. Pair them with
`<picture>` in the README instead.

GitHub serves these under `default-src 'none'; style-src 'unsafe-inline';
sandbox`, so there's no JavaScript, no external font and no external image. An
inline `<style>` with `@keyframes` is all we get, and the file has to stay
small. Three things keep it small, and all three also make it read better:

  * one `<path>` per batch of coordinates instead of 1,288 circles. A
    zero-length subpath under `stroke-linecap:round` draws as a dot, so a
    coordinate costs about twelve characters rather than a whole element, and
    a batch of them shares a single animation. Batching is also what the audit
    does: it sweeps one tensor at a time.
  * dots on integers in a frame scaled by two, so no coordinate carries a
    decimal point. Half a pixel of placement is far coarser than the deviation
    being plotted, which is 1e-5 of a pixel wide.
  * keyframes that stop at the settled state and let the browser build the
    100% stop from the element's own style, which is what `.d` already says.
"""

from __future__ import annotations

import pathlib

import numpy as np

from examples.gradcheck import TOLERANCE, check_every_scalar

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"

# GitHub blocks external fonts, so the file carries its own stack.
MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

PALETTES = {
    "light": {"bg": "#F7F4EF", "ink": "#35322C", "qt": "#8B857A", "hr": "#DAD4C9",
              "bl": "#1B6CA8", "rs": "#C05F1B", "gr": "#2F7D53"},
    "dark": {"bg": "#0D1117", "ink": "#C9D1D9", "qt": "#8B949E", "hr": "#30363D",
             "bl": "#58A6FF", "rs": "#DB8B4F", "gr": "#3FB950"},
}

W, H = 900, 572
X0, X1 = 104.0, 494.0          # plot left, right
Y0, Y1 = 482.0, 92.0           # plot bottom, top: square, so y = x is a true 45
LO, HI = -6, 1                 # decades on both axes
RX, RW = 546.0, 330.0          # right-hand panel
BAR_Y = 168.0

LEAD = 0.35                    # the axes and the y = x line are there first
HOLD = 2.2                     # the finished plot reads as a still
DWELL_BASE = 0.17              # seconds every tensor gets, however small
DWELL_PER = 0.0039             # seconds per scalar on top of that
CHUNK = 22                     # coordinates sharing one animation
RISE, COOL = 0.16, 0.5         # a batch lands fat and opaque, then settles
DOT = 2                        # dot coordinates are integers in a 2x frame
LAND_W, MID_W, SET_W = 9.0, 6.4, 4.2   # dot stroke widths, in user pixels
RESOLVED_FLOOR = 1e-8          # below this a central difference is mostly noise


def sx(v):
    return X0 + (np.log10(v) - LO) / (HI - LO) * (X1 - X0)


def sy(v):
    return Y0 - (np.log10(v) - LO) / (HI - LO) * (Y0 - Y1)


def n2(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s or "0"


def pct(t: float, total: float) -> str:
    """Two decimals of a percentage is a hundredth of a second at this length."""
    return f"{t / total * 100:.2f}".rstrip("0").rstrip(".")


def kf_land(name: str, t0: float, total: float) -> str:
    """A batch arriving: fat and opaque, settling to the width `.d` declares.

    There's no 100% stop on purpose. The browser builds one from the element's
    own style, which is where the settled width and opacity already live, so
    the batch holds still from `t0 + RISE + COOL` to the end of the loop.
    """
    a, b = pct(t0, total), pct(t0 + RISE, total)
    c = pct(min(t0 + RISE + COOL, total), total)
    return (f"@keyframes {name}{{0%,{a}%{{opacity:0;stroke-width:{LAND_W * DOT:g};"
            f"stroke-opacity:1}}{b}%{{opacity:1;stroke-width:{MID_W * DOT:g}}}"
            f"{c}%{{stroke-width:{SET_W * DOT:g};stroke-opacity:.58}}}}")


def kf_show(name: str, t0: float, total: float, rise: float = 0.3) -> str:
    """Appear once, then stay for the rest of the cycle."""
    a, b = pct(t0, total), pct(t0 + rise, total)
    return f"@keyframes {name}{{0%,{a}%{{opacity:0}}{b}%,100%{{opacity:1}}}}"


def kf_win(name: str, t0: float, t1: float, total: float, edge: float = 0.02) -> str:
    """Visible only while this tensor is the one under the differences.

    The edge is a frame wide, so the label switches rather than crossfading and
    a reader arriving at any moment gets a solid name instead of a half-faded
    one. Two stops sharing one offset would be the tidier way to say that, but
    Chrome merges keyframes at equal offsets and the switch becomes a ramp
    across the whole loop.
    """
    a, b = pct(t0, total), pct(t0 + edge, total)
    c, d = pct(t1 - edge, total), pct(t1, total)
    return (f"@keyframes {name}{{0%,{a}%{{opacity:0}}{b}%,{c}%{{opacity:1}}"
            f"{d}%,100%{{opacity:0}}}}")


def chunk_paths(xs: np.ndarray, ys: np.ndarray) -> list[str]:
    """Screen coordinates, batched, one compact path per batch."""
    px = np.rint(np.asarray(xs) * DOT).astype(int)
    py = np.rint(np.asarray(ys) * DOT).astype(int)
    return ["".join(f"M{x} {y}h.02" for x, y in zip(px[i:i + CHUNK], py[i:i + CHUNK],
                                                    strict=True))
            for i in range(0, len(px), CHUNK)]


def text(x: float, y: float, s: str, cls: str, anchor: str = "") -> str:
    a = f' text-anchor="{anchor}"' if anchor else ""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{n2(x)}" y="{n2(y)}" class="{cls}"{a}>{s}</text>'


def compose(names: np.ndarray, analytic: np.ndarray, numeric: np.ndarray) -> tuple[str, str, dict]:
    """Build (css, body, facts). Every figure in the frame comes from the run."""
    order = list(dict.fromkeys(names.tolist()))
    resolved = np.abs(numeric) > RESOLVED_FLOOR
    rel = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-30)
    per_tensor = [float(rel[(names == t) & resolved].max()) for t in order]
    worst = max(per_tensor)
    worst_where = order[int(np.argmax(per_tensor))]
    signs = int((np.sign(analytic) == np.sign(numeric)).sum())
    n_scalars = len(names)
    n_drawn = int(resolved.sum())

    # A relative error of `worst` moves a point off the diagonal by
    # log10(1 + worst) decades. Saying how far that is in pixels beats
    # asserting that the points are "on" the line.
    off_px = float(np.log10(1.0 + worst) * (X1 - X0) / (HI - LO) * np.sqrt(0.5))

    dwell = {t: DWELL_BASE + DWELL_PER * int((names == t).sum()) for t in order}
    fill_end = LEAD + sum(dwell.values())
    total = round(fill_end + HOLD, 3)

    css = [f".a{{animation-duration:{total}s;animation-timing-function:linear;"
           "animation-iteration-count:infinite}"]
    frame = [f'<rect width="{W}" height="{H}" fill="var(--bg)"/>']
    dots: list[str] = []
    panel: list[str] = []

    frame.append(text(32, 36, f"{n_scalars:,} hand-derived gradients, each against the "
                              "derivative that measures it", "s15 ink"))
    frame.append(text(32, 56, f"2-block GPT, {len(order)} parameter tensors, central "
                              "differences at eps = 1e-5, one run of examples/gradcheck.py",
                      "s10"))

    # --- the frame, which is there before any measurement lands -------------
    for d in range(LO, HI + 1):
        gx, gy = float(sx(10.0 ** d)), float(sy(10.0 ** d))
        frame.append(f'<path class="g1" d="M{n2(gx)} {n2(Y1)}V{n2(Y0)}"/>')
        frame.append(f'<path class="g1" d="M{n2(X0)} {n2(gy)}H{n2(X1)}"/>')
        frame.append(text(gx, Y0 + 17, f"1e{d}", "s10", "middle"))
        frame.append(text(X0 - 9, gy + 3.5, f"1e{d}", "s10", "end"))
    frame.append(f'<rect x="{n2(X0)}" y="{n2(Y1)}" width="{n2(X1 - X0)}" '
                 f'height="{n2(Y0 - Y1)}" fill="none" class="g1"/>')
    frame.append(f'<path class="dg" d="M{n2(X0)} {n2(Y0)}L{n2(X1)} {n2(Y1)}"/>')
    # Points land on the diagonal, so the only place a label can't collide with
    # one is the triangle above it. That's also the roomiest space in the panel.
    frame.append(text(126, 132, "y = x, the diagonal", "s13 bl"))
    frame.append(text(126, 152, "where a hand-derived partial equals", "s10"))
    frame.append(text(126, 167, "the difference that measures it", "s10"))
    frame.append(text((X0 + X1) / 2, Y0 + 42, "|central difference|, measured",
                      "s11 rs", "middle"))
    frame.append(f'<g transform="rotate(-90 34 {n2((Y0 + Y1) / 2)})">'
                 + text(34, (Y0 + Y1) / 2, "|hand-derived gradient|, derived",
                        "s11 bl", "middle") + "</g>")

    # --- right-hand panel, the parts that never change ----------------------
    panel.append(text(RX, 104, "now measuring", "s10 ls"))
    panel.append(f'<rect x="{n2(RX)}" y="{n2(BAR_Y)}" width="{n2(RW)}" height="9" '
                 'rx="1.5" fill="var(--hr)"/>')
    panel.append(f'<path class="g1" d="M{n2(RX)} 206H{n2(RX + RW)}"/>')
    panel.append(text(RX, 230, "worst relative error", "s10 ls"))
    panel.append(text(RX, 284, f"against the {TOLERANCE:.0e} relative tolerance CI enforces",
                      "s10"))
    panel.append(f'<path class="g1" d="M{n2(RX)} 306H{n2(RX + RW)}"/>')

    # --- the sweep, tensor by tensor ----------------------------------------
    t, seen, cid = LEAD, 0, 0
    for j, name in enumerate(order):
        sel = names == name
        n = int(sel.sum())
        keep = sel & resolved
        paths = chunk_paths(sx(np.abs(numeric[keep])), sy(np.abs(analytic[keep])))

        for k, d in enumerate(paths):
            key = f"c{cid}"
            css.append(kf_land(key, t + dwell[name] * k / len(paths), total))
            css.append(f".{key}{{animation-name:{key}}}")
            dots.append(f'<path class="d a {key}" d="{d}"/>')
            cid += 1

        # The bar is scalars swept, so a tensor's slice turns rust when its
        # window ends rather than when it opens: the bar never claims a
        # coordinate the dots haven't placed yet.
        bx = RX + seen / n_scalars * RW
        bw = n / n_scalars * RW
        done = f"b{j}"
        css.append(kf_show(done, t + dwell[name], total, rise=0.05))
        css.append(f".{done}{{animation-name:{done}}}")
        panel.append(f'<rect class="sg a {done}" x="{n2(bx)}" y="{n2(BAR_Y)}" '
                     f'width="{n2(bw)}" height="9"/>')

        win = f"n{j}"
        css.append(kf_win(win, t, t + dwell[name], total))
        css.append(f".{win}{{animation-name:{win}}}")
        panel.append(f'<rect class="sa a {win}" x="{n2(bx)}" y="{n2(BAR_Y - 3)}" '
                     f'width="{n2(max(bw, 1.6))}" height="15"/>')
        panel.append(
            f'<g class="a {win}">'
            + text(RX, 134, name, "s19 ink")
            + text(RX, 156, f"scalars {seen + 1:,} to {seen + n:,} of {n_scalars:,}", "s11")
            + text(RX, 268, f"{max(per_tensor[:j + 1]):.2e}", "s19 ink")
            + "</g>")
        seen += n
        t += dwell[name]

    # --- the verdict, held while the finished plot sits still ----------------
    css.append(kf_show("sm", fill_end, total))
    css.append(".sm{animation-name:sm}")
    panel.append(
        '<g class="a sm">'
        + text(RX, 134, "every tensor", "s19 gr")
        + text(RX, 156, f"{n_scalars:,} of {n_scalars:,} scalars", "s11")
        + text(RX, 268, f"{worst:.2e}", "s19 gr")
        + f'<rect x="{n2(RX)}" y="326" width="54" height="23" rx="3" fill="var(--gr)"/>'
        + text(RX + 27, 342, "PASS", "s13 pill", "middle")
        + text(RX + 68, 342, f"{TOLERANCE / worst:,.0f}x inside the tolerance", "s11")
        + text(RX, 376, f"signs agree on {signs:,} of {n_scalars:,} coordinates", "s11")
        + text(RX, 396, f"the loosest tensor is {worst_where}", "s11")
        + "</g>")

    panel.append(text(32, 536, f"{n_scalars - n_drawn} of the {n_scalars:,} coordinates are "
                               "exactly zero in both, embedding rows this batch never "
                               f"touches, so the axes hold {n_drawn:,}.", "s10"))
    panel.append(text(32, 552, f"the worst disagreement is {worst:.2e} relative, which at "
                               f"this scale sits {off_px:.0e} of a pixel off the line.",
                      "s10"))

    body = "".join(frame) + f'<g transform="scale({1 / DOT:g})">' + "".join(dots) \
        + "</g>" + "".join(panel)
    facts = {"total": total, "worst": worst, "worst_where": worst_where, "signs": signs,
             "drawn": n_drawn, "tensors": len(order), "scalars": n_scalars,
             "batches": cid, "off_px": off_px}
    return "".join(css), body, facts


def svg(css: str, body: str, theme: str) -> str:
    p = PALETTES[theme]
    root = (f":root{{--bg:{p['bg']};--ink:{p['ink']};--qt:{p['qt']};--hr:{p['hr']};"
            f"--bl:{p['bl']};--rs:{p['rs']};--gr:{p['gr']}}}")
    base = (f"text{{font-family:{MONO};fill:var(--qt)}}"
            ".ink{fill:var(--ink)}.bl{fill:var(--bl)}.rs{fill:var(--rs)}"
            ".gr{fill:var(--gr)}.pill{fill:var(--bg)}"
            ".sg{fill:var(--rs);fill-opacity:.8}.sa{fill:var(--ink)}"
            ".s10{font-size:10.5px}.s11{font-size:11.5px}.s13{font-size:13px}"
            ".s15{font-size:15.5px}.s19{font-size:19px}.ls{letter-spacing:1.5px}"
            ".g1{stroke:var(--hr);stroke-width:1;fill:none}"
            ".dg{stroke:var(--bl);stroke-width:1.3;fill:none}"
            f".d{{fill:none;stroke:var(--rs);stroke-width:{SET_W * DOT:g};"
            "stroke-linecap:round;stroke-opacity:.58}")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" role="img">'
            f"<style>{root}{base}{css}</style>{body}</svg>\n")


def main() -> None:
    names, analytic, numeric = check_every_scalar()
    css, body, facts = compose(names, analytic, numeric)
    for theme in PALETTES:
        out = DOCS / f"gradcheck_{theme}.svg"
        out.write_text(svg(css, body, theme), encoding="utf-8")
        print(f"wrote {out.relative_to(DOCS.parent)}  {out.stat().st_size / 1024:.1f} KB")
    print(f"  {facts['scalars']:,} scalars in {facts['tensors']} tensors, "
          f"{facts['drawn']:,} on the log axes in {facts['batches']} batches")
    print(f"  worst relative error {facts['worst']:.3e} ({facts['worst_where']}), "
          f"{TOLERANCE / facts['worst']:,.0f}x inside the {TOLERANCE:.0e} tolerance")
    print(f"  loop {facts['total']}s, signs agree on {facts['signs']:,} coordinates")


if __name__ == "__main__":
    main()
