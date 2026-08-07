"""The driver the browser demo calls into.

Everything on the page comes from this file calling `tfs`. The page holds no
model, no gradients and no cached numbers; it draws what these functions
return. That is the whole claim the page makes about itself, so the rule here
is that nothing may be precomputed and nothing may be approximated for speed.

Imports are limited to the standard library, numpy and `tfs`, because Pyodide
has exactly those.
"""

from __future__ import annotations

import random
import time

import numpy as np

import tfs
import tfs.layers as layers
import tfs.model as model
import tfs.ops as ops
from tfs import GPT, AdamLite
from tfs.ops import softmax_crossentropy

# The same fixture examples/gradcheck.py audits, so the numbers on the page
# and the numbers in the README come from one model on one batch.
IDS = np.array([[1, 2, 1, 2, 5], [3, 3, 0, 1, 1]])
TGT = np.array([[2, 1, 2, 5, 6], [3, 0, 1, 1, 4]])
TOLERANCE = 1e-4
EPS = 1e-5

# A pure relative test is the wrong instrument near zero. The central
# difference has an absolute noise floor of roughly eta*|f|/eps, about 4e-11
# here, so a coordinate whose true gradient is that small can agree perfectly
# in absolute terms and still show a relative error of order one. Judging it
# relatively marks a correct derivative red, which is what happened on inputs
# as ordinary as "1 0 4". Pass on either test, the way numpy.allclose does.
ATOL = 1e-9


VOCAB, MAX_T = 7, 6

# The architecture the reader is auditing. Everything downstream reads this at
# call time, so changing it re-points the whole page: a different parameter
# count, different tensors, a different model to be right or wrong about.
ARCH = {"d_model": 8, "n_heads": 2, "n_blocks": 2, "d_ff": 16, "seed": 0}

# Above this, auditing every scalar stops being interactive. The cost is two
# forward passes per scalar and the forward pass itself grows with d_model, so
# it climbs faster than the parameter count does. CI has the same problem and
# solves it the same way, by sampling inside the large tensors.
FULL_AUDIT_MAX = 5000


def build_model() -> GPT:
    return GPT(vocab_size=VOCAB, d_model=ARCH["d_model"],
               n_heads=ARCH["n_heads"], d_ff=ARCH["d_ff"],
               n_blocks=ARCH["n_blocks"], max_T=MAX_T, seed=ARCH["seed"])


def set_arch(d_model=None, n_heads=None, n_blocks=None, d_ff=None, seed=None):
    """Rebuild the model the reader is auditing."""
    want = dict(ARCH)
    for k, v in (("d_model", d_model), ("n_heads", n_heads),
                 ("n_blocks", n_blocks), ("d_ff", d_ff), ("seed", seed)):
        if v is not None:
            want[k] = int(v)
    if want["d_model"] % want["n_heads"]:
        raise ValueError(
            f"{want['n_heads']} heads do not divide d_model {want['d_model']}")
    if not 1 <= want["n_blocks"] <= 4:
        raise ValueError("blocks must be 1 to 4")
    ARCH.update(want)
    return arch_info()


def arch_info():
    model = build_model()
    n = sum(p.data.size for p in model.params())
    return {**ARCH, "params": int(n),
            "tensors": len(model.named_params()),
            "head_dim": ARCH["d_model"] // ARCH["n_heads"],
            "full_audit": n <= FULL_AUDIT_MAX,
            "limit": FULL_AUDIT_MAX}


def set_batch(rows):
    """Audit against a batch the reader chose.

    Every function below reads IDS and TGT at call time, so replacing them
    here re-points the whole page: different data, different loss, different
    gradients, and the same 1,312 checks. Targets are next-token, so a row of
    n ids needs n+1 tokens and the last one is only ever a target.
    """
    global IDS, TGT
    if not rows:
        raise ValueError("give me at least one row")
    clean = []
    for row in rows:
        toks = [int(t) for t in row]
        if len(toks) < 2:
            raise ValueError("a row needs at least 2 tokens to have a target")
        if any(not 0 <= t < VOCAB for t in toks):
            raise ValueError(f"tokens must be 0 to {VOCAB - 1}")
        clean.append(toks[:MAX_T + 1])
    width = min(len(r) for r in clean)
    IDS = np.array([r[:width - 1] for r in clean])
    TGT = np.array([r[1:width] for r in clean])
    return batch_info()


def reset_batch():
    """Back to the fixture examples/gradcheck.py and the README both use."""
    global IDS, TGT
    IDS = np.array([[1, 2, 1, 2, 5], [3, 3, 0, 1, 1]])
    TGT = np.array([[2, 1, 2, 5, 6], [3, 0, 1, 1, 4]])
    return batch_info()


def batch_info():
    model = build_model()
    for p in model.params():
        p.zero_grad()
    model.loss_and_grads(IDS, TGT)
    grads = np.concatenate([p.grad.ravel() for _, p in model.named_params()])
    return {"ids": IDS.tolist(), "targets": TGT.tolist(),
            "rows": int(IDS.shape[0]), "width": int(IDS.shape[1]),
            "loss": _loss(model), "vocab": VOCAB, "max_T": MAX_T,
            "grad_norm": float(np.linalg.norm(grads)),
            "n_grads": int(grads.size)}


def _loss(model: GPT) -> float:
    logits, _ = model.forward(IDS)
    value, _ = softmax_crossentropy(logits, TGT)
    return float(value)


# ---------------------------------------------------------------------------
# sabotage: the point of section 03
#
# These are the mistakes a hand derivation actually makes, not random noise
# injected to make a chart move. Each one is a term someone forgot, and each
# is invisible in the forward pass, which is exactly why the check exists.
# ---------------------------------------------------------------------------

_TRUE_LN = ops.layernorm_backward
_TRUE_GELU = ops.gelu_backward


def _ln_no_covariance(d_out, cache):
    """LayerNorm backward missing its third term.

    x_hat depends on the variance, which depends on every coordinate, so
    differentiating it produces a term in x_hat * sum(d_x_hat * x_hat).
    Treating the normalisation as a constant scale drops it.
    """
    x_hat, gamma, inv = cache
    N = x_hat.shape[-1]
    d_gamma = (d_out * x_hat).sum(axis=tuple(range(d_out.ndim - 1)))
    d_beta = d_out.sum(axis=tuple(range(d_out.ndim - 1)))
    d_x_hat = d_out * gamma
    d_x = (1.0 / N) * inv * (N * d_x_hat - d_x_hat.sum(axis=-1, keepdims=True))
    return d_x, d_gamma, d_beta


def _ln_no_mean(d_out, cache):
    """LayerNorm backward missing its mean-subtraction term.

    Subtracting the mean is part of the op, so its derivative is part of the
    gradient. This keeps the variance term and drops the mean one.
    """
    x_hat, gamma, inv = cache
    N = x_hat.shape[-1]
    d_gamma = (d_out * x_hat).sum(axis=tuple(range(d_out.ndim - 1)))
    d_beta = d_out.sum(axis=tuple(range(d_out.ndim - 1)))
    d_x_hat = d_out * gamma
    d_x = (1.0 / N) * inv * (
        N * d_x_hat - x_hat * (d_x_hat * x_hat).sum(axis=-1, keepdims=True)
    )
    return d_x, d_gamma, d_beta


def _gelu_tanh_only(x, d_out):
    """GELU backward keeping only the cdf term.

    d/dx [x * Phi(x)] is Phi(x) + x * phi(x). Reading GELU as "scale by a
    gate" and differentiating only the scale loses the second term.
    """
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    return d_out * 0.5 * (1.0 + np.tanh(inner))


BUGS = {
    "none": ("the derivation as written", None, None),
    "ln_cov": ("LayerNorm backward, covariance term dropped",
               "layernorm_backward", _ln_no_covariance),
    "ln_mean": ("LayerNorm backward, mean term dropped",
                "layernorm_backward", _ln_no_mean),
    "gelu": ("GELU backward, x * phi(x) term dropped",
             "gelu_backward", _gelu_tanh_only),
}

_active = "none"


# Every module that did `from .ops import ...` holds its own binding, and all
# of them have to move together. Enumerating them by hand is how this went
# wrong once already: layers.py and ops.py were patched, model.py was not, so
# the final LayerNorm kept the correct backward and the audit reported 20 of
# 29 tensors wrong where a real derivation mistake gives 26. Worse, the six
# tensors that read clean did have gradients flowing through the broken op,
# which made the page's explanation of the failure pattern false. Discover the
# holders instead of listing them.
_HOLDERS = (ops, layers, model, tfs)


def set_bug(kind: str) -> str:
    """Swap a wrong backward pass in, or put the right one back."""
    global _active
    if kind not in BUGS:
        raise ValueError(f"unknown bug {kind!r}")
    for name, true in (("layernorm_backward", _TRUE_LN),
                       ("gelu_backward", _TRUE_GELU)):
        for mod in _HOLDERS:
            if hasattr(mod, name):
                setattr(mod, name, true)
    _, target, fn = BUGS[kind]
    if fn is not None:
        for mod in _HOLDERS:
            if hasattr(mod, target):
                setattr(mod, target, fn)
    _active = kind
    return kind


def bug_list():
    return [{"key": k, "label": v[0]} for k, v in BUGS.items()]


# ---------------------------------------------------------------------------
# 01: one derivative, from the definition
# ---------------------------------------------------------------------------

def tensor_list():
    """Every parameter tensor, with its shape and its values."""
    model = build_model()
    for prm in model.params():
        prm.zero_grad()
    model.loss_and_grads(IDS, TGT)
    out = []
    for name, p in model.named_params():
        # the gradient is what the page is about, and a coordinate the batch
        # never touches has none, so ship both and let the grid show which
        # cells are actually live
        out.append({"name": name, "shape": list(p.data.shape),
                    "size": int(p.data.size),
                    "values": [float(v) for v in p.data.flat],
                    "grads": [float(v) for v in p.grad.flat]})
    return {"tensors": out, "total": sum(t["size"] for t in out)}


def one_derivative(name: str, index: int, eps: float = EPS):
    """The full audit of a single scalar, with both sides shown.

    Returns the analytic gradient the hand-written backward pass produced and
    the two forward passes that measure it, so the page can show the
    subtraction rather than just its result.
    """
    model = build_model()
    for p in model.params():
        p.zero_grad()
    model.loss_and_grads(IDS, TGT)
    named = dict(model.named_params())
    if name not in named:
        raise ValueError(f"no parameter tensor {name!r}")
    param = named[name]
    if not 0 <= index < param.data.size:
        raise ValueError(f"{name} has {param.data.size} scalars")

    analytic = float(param.grad.flat[index])
    original = float(param.data.flat[index])
    param.data.flat[index] = original + eps
    up = _loss(model)
    param.data.flat[index] = original - eps
    down = _loss(model)
    param.data.flat[index] = original
    central = (up - down) / (2 * eps)
    abs_err = abs(analytic - central)
    rel_err = abs_err / max(abs(central), 1e-12)
    return {
        "name": name, "index": index, "shape": list(param.data.shape),
        "theta": original, "eps": eps,
        "analytic": analytic, "up": up, "down": down,
        "central": central, "diff": up - down,
        "abs_err": abs_err, "rel_err": rel_err, "atol": ATOL,
        "tiny": abs_err <= ATOL and rel_err >= TOLERANCE,
        "passes": abs_err <= ATOL or rel_err < TOLERANCE,
        "bug": _active,
    }


# ---------------------------------------------------------------------------
# 02: all 1,312, in chunks so the page can draw as it goes
# ---------------------------------------------------------------------------

_sweep_state: dict = {}


def audit_begin(eps: float = EPS, per_tensor: int = 0):
    """One backward pass fills every .grad; after this we only perturb.

    `per_tensor` caps how many coordinates are measured inside each tensor.
    Zero means every scalar. Sampling is what makes a larger model auditable
    while you wait, and it is what CI already does on the big tensors, so the
    page reports which mode it ran in rather than quietly doing less work.
    """
    model = build_model()
    for p in model.params():
        p.zero_grad()
    model.loss_and_grads(IDS, TGT)
    rng = random.Random(12345)
    coords, total = [], 0
    for name, param in model.named_params():
        size = param.data.size
        total += size
        if per_tensor and size > per_tensor:
            # spread over the tensor rather than taking a prefix, which would
            # only ever look at one row of a weight matrix
            picks = sorted(rng.sample(range(size), per_tensor))
        else:
            picks = range(size)
        coords.extend((name, c) for c in picks)
    _sweep_state.clear()
    _sweep_state.update(model=model, named=dict(model.named_params()),
                        coords=coords, i=0, eps=float(eps),
                        t0=time.perf_counter(), rows=[])
    return {"total": len(coords), "scalars": total, "eps": float(eps),
            "sampled": len(coords) < total, "per_tensor": per_tensor,
            "tensors": [n for n, _ in model.named_params()], "bug": _active}


def audit_step(n: int = 160):
    """Measure the next n coordinates. Returns them and the running worst."""
    st = _sweep_state
    if not st:
        raise RuntimeError("call audit_begin first")
    eps, named = st["eps"], st["named"]
    out = []
    end = min(st["i"] + n, len(st["coords"]))
    for k in range(st["i"], end):
        name, c = st["coords"][k]
        param = named[name]
        analytic = float(param.grad.flat[c])
        original = float(param.data.flat[c])
        param.data.flat[c] = original + eps
        up = _loss(st["model"])
        param.data.flat[c] = original - eps
        down = _loss(st["model"])
        param.data.flat[c] = original
        central = (up - down) / (2 * eps)
        err = abs(analytic - central)
        rel = err / max(abs(central), 1e-12)
        # below the difference quotient's own noise floor there is nothing
        # left to measure, so score it on the absolute gap instead
        out.append({"name": name, "i": c, "a": analytic, "n": central,
                    "r": 0.0 if err <= ATOL else rel})
    st["i"] = end
    st["rows"].extend(out)
    done = st["i"] >= len(st["coords"])
    worst = max((r["r"] for r in st["rows"]), default=0.0)
    result = {"rows": out, "done": done, "at": st["i"],
              "total": len(st["coords"]), "worst": worst}
    if out:
        w = max(out, key=lambda r: r["r"])
        result["worst_now"] = {"name": w["name"], "i": w["i"], "r": w["r"]}
    if done:
        result["ms"] = (time.perf_counter() - st["t0"]) * 1000.0
        per = {}
        for r in st["rows"]:
            if r["r"] > per.get(r["name"], -1.0):
                per[r["name"]] = r["r"]
        result["per_tensor"] = [{"name": k, "worst": v} for k, v in per.items()]
        result["failing"] = sorted(k for k, v in per.items() if v > TOLERANCE)
        result["tolerance"] = TOLERANCE
    return result


# ---------------------------------------------------------------------------
# 04: why eps = 1e-5
# ---------------------------------------------------------------------------

def eps_curve_point(name: str, index: int, eps: float):
    """One coordinate measured at one eps, cheap enough for a live slider."""
    r = one_derivative(name, index, eps)
    return {"eps": eps, "rel_err": r["rel_err"], "central": r["central"],
            "analytic": r["analytic"], "diff": r["diff"]}


def eps_point(eps: float, stride: int = 7):
    """Median relative error at one eps, on every stride-th coordinate.

    Striding keeps a nine-point sweep interactive. The V it traces is the
    same shape the full sweep gives, because the median is not sensitive to
    which representative subset it is taken over.
    """
    model = build_model()
    for p in model.params():
        p.zero_grad()
    model.loss_and_grads(IDS, TGT)
    rels = []
    k = 0
    for _, param in model.named_params():
        for c in range(param.data.size):
            k += 1
            if k % stride:
                continue
            analytic = float(param.grad.flat[c])
            original = float(param.data.flat[c])
            param.data.flat[c] = original + eps
            up = _loss(model)
            param.data.flat[c] = original - eps
            down = _loss(model)
            param.data.flat[c] = original
            central = (up - down) / (2 * eps)
            rels.append(abs(analytic - central) / max(abs(central), 1e-12))
    rels.sort()
    mid = len(rels) // 2
    median = rels[mid] if len(rels) % 2 else (rels[mid - 1] + rels[mid]) / 2
    return {"eps": eps, "median": median, "n": len(rels)}


# ---------------------------------------------------------------------------
# 00 and 03: training, with or without the bug in
# ---------------------------------------------------------------------------

PERIOD = [1, 2, 3, 4, 5]


def set_period(tokens):
    """The sequence the model is asked to learn.

    A shorter period fits inside the context window with room to spare; a
    longer one does not, which is the whole of experiment five.
    """
    global PERIOD
    toks = [int(t) for t in tokens]
    if not 2 <= len(toks) <= 8:
        raise ValueError("a period of 2 to 8 tokens")
    if any(not 0 <= t < VOCAB for t in toks):
        raise ValueError(f"tokens must be 0 to {VOCAB - 1}")
    PERIOD = toks
    return {"period": PERIOD, "length": len(PERIOD), "window": MAX_T}


def _batch(rng, batch: int, T: int, fixed_offset: bool):
    ids = np.zeros((batch, T), dtype=int)
    tgt = np.zeros((batch, T), dtype=int)
    for b in range(batch):
        off = 0 if fixed_offset else int(rng.integers(len(PERIOD)))
        seq = [PERIOD[(off + i) % len(PERIOD)] for i in range(T + 1)]
        ids[b] = seq[:T]
        tgt[b] = seq[1:]
    return ids, tgt


_train_state: dict = {}


def train_begin(lr: float = 5e-3, batch: int = 8, seed: int = 0,
                fixed_offset: bool = False):
    model = GPT(vocab_size=7, d_model=8, n_heads=2, d_ff=16,
                n_blocks=2, max_T=6, seed=seed)
    _train_state.clear()
    _train_state.update(model=model, opt=AdamLite(model.params(), lr=lr),
                        rng=np.random.default_rng(seed), batch=batch,
                        fixed=fixed_offset, step=0, losses=[])
    return {"bug": _active, "lr": lr, "batch": batch}


def train_step(n: int = 20):
    st = _train_state
    if not st:
        raise RuntimeError("call train_begin first")
    model, opt = st["model"], st["opt"]
    T = model.max_T
    for _ in range(n):
        ids, tgt = _batch(st["rng"], st["batch"], T, st["fixed"])
        opt.zero_grad()
        loss = model.loss_and_grads(ids, tgt)
        opt.step()
        st["step"] += 1
        st["losses"].append(float(loss))
    return {"step": st["step"], "losses": st["losses"][-n:],
            "last": st["losses"][-1]}


def train_sample(prompt=(1, 2, 3), max_new: int = 7, temperature: float = 0.0):
    st = _train_state
    if not st:
        raise RuntimeError("call train_begin first")
    out = st["model"].generate(np.array(list(prompt)), max_new=max_new,
                               temperature=float(temperature))
    got = [int(v) for v in out[len(prompt):]]
    # What continues the sequence depends on the last token the reader gave,
    # not on how many they gave. Keying off the length assumes every prompt
    # starts at phase 0, which marks a correct continuation from any other
    # phase as wrong.
    last = int(prompt[-1])
    if last in PERIOD:
        at = PERIOD.index(last)
        want = [PERIOD[(at + 1 + i) % len(PERIOD)] for i in range(max_new)]
        correct = got == want
    else:
        # 0 and 6 never appear in the training sequence, so there is no
        # continuation to be right or wrong about.
        want, correct = None, None
    return {"prompt": list(prompt), "out": [int(v) for v in out],
            "continuation": got, "expected": want, "correct": correct,
            "temperature": float(temperature), "period": list(PERIOD),
            "in_period": last in PERIOD, "step": st["step"], "bug": _active}


def grad_compare(kind: str):
    """How far a wrong derivation's gradient points from the right one.

    The answer for every bug here is "barely", and that is the finding. A
    gradient can be wrong by two orders of magnitude on individual
    coordinates and still sit within a couple of degrees of the true one,
    because the error is spread thinly over 1,312 of them. Adam then
    normalises per coordinate and descends anyway. This is why a loss curve
    cannot audit a derivative.
    """
    was = _active

    def snapshot(bug):
        set_bug(bug)
        model = build_model()
        for p in model.params():
            p.zero_grad()
        model.loss_and_grads(IDS, TGT)
        return np.concatenate([p.grad.ravel() for _, p in model.named_params()])

    true = snapshot("none")
    other = snapshot(kind)
    set_bug(was)
    nt, no = np.linalg.norm(true), np.linalg.norm(other)
    cos = float(true @ other / (nt * no)) if nt and no else 0.0
    cos = max(-1.0, min(1.0, cos))
    return {"bug": kind, "cosine": cos,
            "degrees": float(np.degrees(np.arccos(cos))),
            "max_abs_diff": float(np.abs(true - other).max()),
            "norm_ratio": float(no / nt) if nt else 0.0}


def train_pair(kind: str, steps: int = 400, seed: int = 0):
    """The same run twice, once with the derivation right and once wrong.

    Returned together so the page can draw them on one axis, where they land
    on top of each other.
    """
    was = _active
    out = {}
    for tag, bug in (("true", "none"), ("bug", kind)):
        set_bug(bug)
        train_begin(seed=seed)
        train_step(steps)
        out[tag] = {"losses": list(_train_state["losses"]),
                    "sample": train_sample()}
    set_bug(was)
    out["steps"] = steps
    out["compare"] = grad_compare(kind)
    return out


def attention(tokens=None, trained: bool = True):
    """Every head's attention matrix, for the page to draw.

    The forward cache already holds these, so this is a read rather than a
    second computation. Untrained heads are close to uniform over whatever the
    causal mask allows; the interesting thing is watching that break as the
    model learns, which is why this takes `trained` rather than always
    building a fresh model.
    """
    toks = [int(t) for t in (tokens or PERIOD[:MAX_T])][:MAX_T]
    if len(toks) < 2:
        raise ValueError("at least 2 tokens")
    if any(not 0 <= t < VOCAB for t in toks):
        raise ValueError(f"tokens must be 0 to {VOCAB - 1}")

    if trained and _train_state:
        model = _train_state["model"]
        step = _train_state["step"]
    else:
        model = build_model()
        step = 0
    logits, cache = model.forward(np.array([toks]))
    _, block_caches, _, _ = cache

    heads = []
    for b, bc in enumerate(block_caches):
        attn = bc[1][4]                      # (B, h, T, T)
        for h in range(attn.shape[1]):
            m = attn[0, h]
            # how far from "spread evenly over everything I am allowed to see"
            T = m.shape[0]
            ent = 0.0
            for i in range(T):
                row = m[i, : i + 1]
                p_ = row[row > 1e-12]
                ent += float(-(p_ * np.log(p_)).sum()) / max(1, T)
            heads.append({"block": b, "head": h,
                          "w": [[float(v) for v in row] for row in m],
                          "entropy": ent})
    probs = np.exp(logits[0, -1] - logits[0, -1].max())
    probs /= probs.sum()
    return {"tokens": toks, "heads": heads, "step": step,
            "n_blocks": len(block_caches),
            "n_heads": len(heads) // max(1, len(block_caches)),
            "next": [float(v) for v in probs]}


# ---------------------------------------------------------------------------
# 05: the position where it stops working
# ---------------------------------------------------------------------------

def positional(steps: int = 400, seed: int = 0, n_pos: int = 14):
    """Per-position greedy accuracy for a model trained both ways.

    The fixed-offset model only ever saw phase 0, so it is exact inside its
    training window and has nothing to say the moment the window slides.
    """
    out = {}
    for tag, fixed in (("fixed", True), ("random", False)):
        train_begin(seed=seed, fixed_offset=fixed)
        train_step(steps)
        model = _train_state["model"]
        seq = [PERIOD[i % len(PERIOD)] for i in range(n_pos + 1)]
        acc = []
        for pos in range(1, n_pos + 1):
            lo = max(0, pos - model.max_T)
            ctx = np.array([seq[lo:pos]])
            logits, _ = model.forward(ctx)
            pred = int(np.argmax(logits[0, -1]))
            acc.append(1.0 if pred == seq[pos] else 0.0)
        out[tag] = acc
    out["positions"] = list(range(1, n_pos + 1))
    out["window"] = MAX_T
    out["period"] = list(PERIOD)
    out["steps"] = steps
    return out
