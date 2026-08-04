"""Regenerate the figures used in the README.

Everything here is computed from `tfs` at run time — there is no stored
data anywhere in this repo, and no figure is drawn from numbers that were
not produced by the code next to it.

    PYTHONPATH=. python3 docs/figures.py

Writes all three README figures:
    docs/gradcheck.png    — every parameter gradient against central
                            differences, per-tensor error, and the step-size
                            sweep that explains why eps = 1e-5.
    docs/kv_cache.png     — decode work with and without the key/value cache.
    docs/positional_generalization.png
                          — fixed- versus random-offset training.

Every input is deterministic — fixed seeds, and counted operations rather
than wall-clock — so re-running this on the same matplotlib writes
byte-identical PNGs. The one number that cannot be made portable, the
decode wall-clock, is printed instead of plotted.

Requires matplotlib: `pip install -e ".[dev]"`.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from examples.gradcheck import IDS, TGT, build_model, check_every_scalar, loss_fn
from tfs.model import GPT

DOCS = pathlib.Path(__file__).resolve().parent

# One colour per family of parameter, reused across panels so the eye can
# carry a tensor from the scatter to the bar chart.
FAMILY_COLOUR = {
    "embedding": "#3b5bdb",
    "attention": "#e8590c",
    "layernorm": "#0b7285",
    "ffn": "#862e9c",
    "lm_head": "#495057",
}


def family_of(name: str) -> str:
    if "attn" in name:
        return "attention"
    if "ffn" in name:
        return "ffn"
    if "ln" in name:
        return "layernorm"
    if "lm_head" in name:
        return "lm_head"
    return "embedding"


def step_size_sweep(per_tensor: int = 8, seed: int = 1):
    """Median relative error of the central difference as a function of eps.

    Two error sources fight here: truncation, which falls like eps^2, and
    floating-point cancellation in (f(x+eps) - f(x-eps)), which grows like
    eta/eps for machine epsilon eta. Their sum is a V with a minimum near
    eta^(1/3) ~ 6e-6.
    """
    model = build_model()
    for p in model.params():
        p.zero_grad()
    model.loss_and_grads(IDS, TGT)
    loss = loss_fn(model)

    rng = np.random.default_rng(seed)
    picks = []
    for _, param in model.named_params():
        n = param.data.size
        for c in rng.choice(n, size=min(n, per_tensor), replace=False):
            if abs(float(param.grad.flat[c])) > 1e-6:
                picks.append((param, int(c)))

    grid = np.logspace(-11, -1, 21)
    median_err = []
    for eps in grid:
        errs = []
        for param, c in picks:
            original = param.data.flat[c]
            param.data.flat[c] = original + eps
            up = loss()
            param.data.flat[c] = original - eps
            down = loss()
            param.data.flat[c] = original
            fd = (up - down) / (2 * eps)
            ad = float(param.grad.flat[c])
            errs.append(abs(ad - fd) / abs(ad))
        median_err.append(float(np.median(errs)))
    return grid, np.array(median_err), len(picks)


def fig_gradcheck(out: pathlib.Path | None = None) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, analytic, numeric = check_every_scalar()
    grid, median_err, n_sweep = step_size_sweep()

    signs_agree = int((np.sign(analytic) == np.sign(numeric)).sum())
    resolved = np.abs(numeric) > 1e-8
    rel = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-30)

    fig = plt.figure(figsize=(14.0, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.92, 1.03],
                          left=0.048, right=0.985, bottom=0.115, top=0.845,
                          wspace=0.42)

    # --- (a) analytic vs finite difference, every scalar ------------------
    ax = fig.add_subplot(gs[0, 0])
    lim = (10 ** np.floor(np.log10(np.abs(numeric[resolved]).min())),
           10 ** np.ceil(np.log10(np.abs(numeric).max())))
    ax.plot(lim, lim, color="#adb5bd", lw=1.0, zorder=1)
    for fam, colour in FAMILY_COLOUR.items():
        sel = np.array([family_of(n) == fam for n in names]) & resolved
        ax.scatter(np.abs(numeric[sel]), np.abs(analytic[sel]), s=7,
                   color=colour, alpha=0.55, linewidths=0, label=fam, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("|central difference|")
    ax.set_ylabel("|hand-derived gradient|")
    ax.set_title("(a)  every scalar, on the diagonal", fontsize=11, loc="left")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False, markerscale=2.2,
              handletextpad=0.2, borderpad=0.1)
    ax.grid(alpha=0.18, which="both")
    ax.text(0.97, 0.05,
            f"{len(names):,} coordinates\n"
            f"{signs_agree:,}/{len(names):,} signs agree\n"
            f"max rel. error {rel[resolved].max():.1e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color="#343a40")

    # --- (b) worst relative error per tensor ------------------------------
    ax = fig.add_subplot(gs[0, 1])
    labels = list(dict.fromkeys(names))
    worst = [rel[(names == t) & resolved].max() for t in labels]
    y = np.arange(len(labels))[::-1]
    colours = [FAMILY_COLOUR[family_of(t)] for t in labels]
    # Lollipops, not bars: on a log axis a bar's left edge is an arbitrary
    # choice of baseline, and these values have no meaningful zero.
    ax.hlines(y, 1e-12, worst, color=colours, lw=1.1, alpha=0.5)
    ax.scatter(worst, y, s=26, color=colours, zorder=3)
    ax.axvline(1e-4, color="#c92a2a", ls="--", lw=1.2)
    ax.text(1e-4, len(labels) + 0.6, " CI tolerance", color="#c92a2a",
            fontsize=8, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("blocks.", "b") for t in labels], fontsize=6.4)
    ax.set_xscale("log")
    ax.set_xlim(1e-12, 3e-3)
    ax.set_ylim(-1.2, len(labels) + 1.4)
    ax.set_xlabel("worst relative error in the tensor")
    ax.set_title(f"(b)  all {len(labels)} parameter tensors",
                 fontsize=11, loc="left")
    ax.grid(alpha=0.18, axis="x", which="both")
    ax.tick_params(axis="y", length=0)

    # --- (c) why eps = 1e-5 ------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    best = int(np.argmin(median_err))
    round_off = median_err[0] * (grid[0] / grid)
    truncation = median_err[-1] * (grid / grid[-1]) ** 2
    ax.plot(grid, round_off, color="#adb5bd", ls=":", lw=1.3)
    ax.plot(grid, truncation, color="#adb5bd", ls=":", lw=1.3)
    ax.plot(grid, median_err, "o-", color="#1864ab", lw=1.8, ms=4.5, zorder=3)
    ax.scatter([grid[best]], [median_err[best]], s=90, facecolor="none",
               edgecolor="#c92a2a", lw=1.6, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 3e-2)
    ax.set_xlabel("finite-difference step $\\varepsilon$")
    ax.set_ylabel("median relative error")
    ax.set_title("(c)  why the step size is $10^{-5}$", fontsize=11, loc="left")
    ax.grid(alpha=0.18, which="both")
    ax.annotate("cancellation\n$\\sim \\eta/\\varepsilon$", xy=(2e-10, 3e-4),
                fontsize=8.5, color="#495057")
    ax.annotate("truncation\n$\\sim \\varepsilon^2$", xy=(6e-3, 2e-6),
                fontsize=8.5, color="#495057", ha="center")
    ax.annotate(f"best at $\\varepsilon$ = {grid[best]:.0e}\n"
                f"median error {median_err[best]:.0e}",
                xy=(grid[best], median_err[best]), xytext=(6.0e-6, 1.25e-10),
                fontsize=8.5, color="#c92a2a", va="bottom", ha="right",
                arrowprops=dict(arrowstyle="-", color="#c92a2a", lw=0.9,
                                shrinkB=8))

    fig.suptitle("Hand-derived gradients vs central differences — "
                 "a 2-block GPT, all 29 parameter tensors",
                 fontsize=13.5, x=0.048, ha="left", y=0.955)
    fig.text(0.048, 0.905,
             "Nothing here is autodiff: every backward pass in tfs/ was "
             "written out by hand and is checked against the definition of "
             "the derivative.",
             fontsize=9.5, color="#495057", ha="left")

    out = out or DOCS / "gradcheck.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}  ({len(names):,} coordinates, "
          f"max rel error {rel[resolved].max():.2e})")
    return out


class _CountingGPT(GPT):
    """A GPT that records how many token-positions it pushes through the stack.

    Wall-clock is the wrong unit for a claim about an algorithm: it moves
    with the machine, the BLAS build, and whatever else the laptop is doing.
    Positions-through-the-stack is a property of the code, and it is the
    same integer everywhere.
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.positions = 0

    def forward(self, ids: np.ndarray):
        self.positions += int(ids.shape[0] * ids.shape[1])
        return super().forward(ids)

    def forward_step(self, ids_new: np.ndarray, pos: int, kv):
        self.positions += int(ids_new.shape[0] * ids_new.shape[1])
        return super().forward_step(ids_new, pos, kv)


def decode_work(lengths=(16, 32, 64, 128, 256), prompt_len: int = 8,
                max_T: int = 512):
    """Token-positions embedded and pushed through the blocks, cached or not.

    Measured by instrumenting `GPT`, then checked against the closed form:
    uncached decoding of n tokens re-runs the whole window every step, for
    sum_i min(prompt_len + i, max_T) positions; cached decoding runs the
    prompt once and one position per step thereafter. If the two ever
    disagree, the assert below fires and the figure does not get drawn.
    """
    lengths = np.asarray(lengths)
    full, cached = [], []
    for n in lengths:
        for use_cache, sink in ((False, full), (True, cached)):
            m = _CountingGPT(vocab_size=32, d_model=16, n_heads=2, d_ff=32,
                             n_blocks=1, max_T=max_T, seed=0)
            m.generate(np.arange(prompt_len), max_new=int(n),
                       temperature=0.0, use_cache=use_cache)
            sink.append(m.positions)
        closed_full = sum(min(prompt_len + i, max_T) for i in range(int(n)))
        assert full[-1] == closed_full, (n, full[-1], closed_full)
        assert cached[-1] == prompt_len + int(n) - 1, (n, cached[-1])
    return lengths, np.array(full), np.array(cached)


def decode_timings(lengths=(16, 32, 64, 128, 256), repeats: int = 2):
    """Wall-clock for the same decode, printed rather than plotted.

    These seconds are the least portable numbers in the repository, which
    is why no figure is drawn from them. They are here because the gap
    between the operation count and the clock is itself informative: a
    cached step still attends over the whole prefix, and at these sizes a
    lot of the remaining time is Python and NumPy call overhead, so the
    speed-up you measure lands well below the ratio of positions.
    """
    model = GPT(vocab_size=256, d_model=128, n_heads=8, d_ff=512,
                n_blocks=4, max_T=512, seed=0)
    prompt = np.arange(8)
    print("wall clock (one machine, one run — expect your own numbers):")
    for n in lengths:
        best = []
        for use_cache in (False, True):
            t = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter()
                model.generate(prompt, max_new=n, temperature=0.0,
                               use_cache=use_cache)
                t = min(t, time.perf_counter() - t0)
            best.append(t)
        print(f"  {n:4d} tokens: full {best[0]:6.3f}s  cached {best[1]:6.3f}s"
              f"  ({best[0] / best[1]:.1f}x)")


def fig_kv_cache(out: pathlib.Path | None = None) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, full, cached = decode_work()
    ratio = full / cached
    slope_full = np.polyfit(np.log(n), np.log(full), 1)[0]
    slope_cached = np.polyfit(np.log(n), np.log(cached), 1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.155, top=0.79,
                        wspace=0.22)

    ax = axes[0]
    ax.plot(n, full, "s--", color="#c92a2a", lw=2, ms=6,
            label=f"full recompute  (slope {slope_full:.2f})")
    ax.plot(n, cached, "o-", color="#1864ab", lw=2, ms=6,
            label=f"key/value cache  (slope {slope_cached:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tokens generated")
    ax.set_ylabel("token-positions through the blocks")
    ax.set_title("(a)  work done to decode", fontsize=11, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(alpha=0.2, which="both")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()

    ax = axes[1]
    ax.bar(np.arange(len(n)), ratio, width=0.55, color="#0b7285", alpha=0.88)
    for i, y in enumerate(ratio):
        ax.annotate(f"{y:.1f}x", (i, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9.5, color="#0b7285")
    ax.set_xticks(np.arange(len(n)))
    ax.set_xticklabels([str(v) for v in n])
    ax.set_xlabel("tokens generated")
    ax.set_ylabel("positions saved (ratio)")
    ax.set_ylim(0, ratio.max() * 1.2)
    ax.set_title("(b)  and what the cache removes", fontsize=11, loc="left")
    ax.grid(alpha=0.2, axis="y")
    ax.set_axisbelow(True)

    fig.suptitle("Incremental decoding: same arithmetic, one order less of it",
                 fontsize=13.5, x=0.055, ha="left", y=0.955)
    fig.text(0.055, 0.885,
             "Counted by instrumenting GPT, not timed — every number here "
             "is an integer and reproduces exactly. Prompt 8 tokens, "
             "max_T 512.",
             fontsize=9.5, color="#495057", ha="left")

    out = out or DOCS / "kv_cache.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}")
    return out


FIXED_COLOUR = "#c92a2a"
RANDOM_COLOUR = "#1864ab"


def fig_positional(positions, acc_fixed, acc_rand, gen_fixed, gen_rand,
                   truth, window: int, out: pathlib.Path | None = None):
    """The positional-generalization tearsheet.

    Lives here rather than in the example that computes it so that all
    three README figures share one set of fonts, colours and panel
    furniture; `examples/positional_generalization.py` does the training
    and hands the arrays over.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.155, top=0.79,
                        wspace=0.22)

    ax = axes[0]
    ax.axvspan(positions[0] - 0.5, window + 0.5, color="#e9ecef", zorder=0)
    ax.text(positions[0] - 0.2, 1.12, "seen in training", fontsize=8.5,
            color="#868e96", va="center", ha="left")
    ax.plot(positions, acc_rand, "o-", color=RANDOM_COLOUR, lw=2, ms=5,
            label="random-offset training", zorder=3)
    ax.plot(positions, acc_fixed, "s--", color=FIXED_COLOUR, lw=2, ms=5,
            label="fixed-offset training (the old demo)", zorder=2)
    ax.axvline(window + 0.5, color="#868e96", ls=":", lw=1.4, zorder=1)
    ax.set_xlabel("absolute position in the period-5 sequence")
    ax.set_ylabel("greedy next-token accuracy")
    ax.set_title("(a)  where the fixed-offset model stops working",
                 fontsize=11, loc="left")
    ax.set_ylim(-0.08, 1.42)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper center", fontsize=9, frameon=False, ncol=2,
              columnspacing=1.4, handletextpad=0.4)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

    ax = axes[1]
    t = np.arange(len(truth))
    ax.axvspan(-0.5, window + 0.5, color="#e9ecef", zorder=0)
    ax.step(t, truth, where="mid", color="#adb5bd", lw=6, alpha=0.55,
            zorder=1, label="the pattern")
    ax.plot(t, gen_rand, "o", color=RANDOM_COLOUR, ms=5.5, zorder=3,
            label="random-offset greedy")
    ax.plot(t, gen_fixed, "s", color=FIXED_COLOUR, ms=5.5, zorder=2,
            mfc="none", mew=1.6, label="fixed-offset greedy")
    ax.axvline(window + 0.5, color="#868e96", ls=":", lw=1.4, zorder=1)
    ax.set_xlabel("absolute position in the generated sequence")
    ax.set_ylabel("token emitted")
    ax.set_yticks(sorted(set(int(v) for v in truth)))
    ax.set_ylim(0.4, 7.0)
    ax.set_title("(b)  what each one actually emits", fontsize=11, loc="left")
    ax.legend(loc="upper center", fontsize=9, frameon=False, ncol=3,
              columnspacing=1.1, handletextpad=0.4)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

    fig.suptitle("Learned absolute positions are a lookup table, "
                 "not a rule",
                 fontsize=13.5, x=0.055, ha="left", y=0.955)
    fig.text(0.055, 0.885,
             f"Same architecture, same steps, same seed — only the training "
             f"offsets differ. The dotted line is position {window + 1}, the "
             f"first prediction whose context window the fixed-offset model "
             f"never saw.",
             fontsize=9.5, color="#495057", ha="left")

    out = out or DOCS / "positional_generalization.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    fig_gradcheck()
    fig_kv_cache()
    from examples.positional_generalization import study
    study()
    decode_timings()


if __name__ == "__main__":
    main()
