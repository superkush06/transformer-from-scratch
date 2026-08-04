# Changelog

## [0.4.2] - 2026-08-03

### Fixed
- `docs/theory.md` said the finite-difference step sweep "samples only
  five steps per decade", and the 0.4.1 entry below repeated it. No grid
  in this repository has that spacing: `step_size_sweep` in
  `docs/figures.py` uses `np.logspace(-11, -1, 21)` — 21 points over ten
  decades, so *two* steps per decade, consecutive points a factor
  sqrt(10) ~ 3.16 apart. The count is load-bearing rather than
  decorative, because that paragraph offers grid coarseness as one of
  two reasons the measured minimum sits 3.6x to the right of eps*: at
  five steps per decade the spacing is 1.58x and quantisation could not
  displace the minimum by more than ~1.3x, so the explanation would not
  hold. At the real spacing it does — the grid point nearest eps* =
  8.7e-06 is 1e-05, one step below the measured minimum at 3.16e-05.
  theory.md now names the spacing and that one-step gap instead of a
  step count nothing computes.
- README bolded the decode speed-up as a **15x to 37x** spread. That
  range is nine *single* attempts, but `decode_timings` reports `min()`
  over two, and the command the paragraph tells you to run prints the
  better-of-two number — which can land above the range advertised right
  next to it. Eight consecutive runs on the machine the README names
  printed 20.3x, 25.5x, 28.5x, 28.8x, 28.8x, 32.2x, 36.5x and 42.8x
  (median 29x; full 3.7–6.8 s, cached 0.13–0.20 s), so a reader
  following the instructions could see 42.8x and find it outside the
  stated spread. Both statistics are now named, with the envelope quoted
  as 15x to 43x. The 132x position count is unchanged — it is an
  operation count, not a clock.

## [0.4.1] - 2026-08-03

### Fixed
- `tests/test_properties.py` drew softmax logits up to |x| ~ 229, and
  shifted ones up to |x + c| ~ 664 — both under the float64 `exp()`
  overflow point of 709.8. The simplex test's docstring claimed to cover
  "logits large enough that a naive exp() overflows", and it did not:
  deleting the max-subtraction from `tfs.ops.softmax` left every softmax
  property test passing. The scale now runs to 1e3, the number of draws
  that actually clear the overflow threshold is asserted rather than
  hoped for, and the shift test checks finiteness before reducing with
  `max()`, which was quietly swallowing NaN rows. Both tests now fail
  with the guard removed.
- `examples/validate.py` hardcoded `agrees=False` for the GELU and
  LayerNorm rows, so the standalone script would have printed `DIFF` and
  `9/11 rungs agree` even after someone made the GELU exact. Both
  verdicts are now derived from the measurement the check just made;
  swapping in `x*Phi(x)` flips row 6 to `ok` on its own. (CI was never at
  risk — the 4e-4..6e-4 band in `tests/test_validation.py` is the real
  guard — but the script's own summary line was not self-checking.)
- README described the worst gradient-check error, 3.1e-07, as "about
  three orders of magnitude inside the tolerance CI enforces". The
  tolerance is the relative 1e-4 in `tests/test_gradcheck.py`, so the
  margin is 330x — two and a half orders, rounded in the flattering
  direction.
- README said the finite-difference sweep bottoms out "where the algebra
  in docs/theory.md says both should be". It does not, quite: the
  measured minimum is 3.6x to the right of the predicted eps* and its
  depth is 11x above the idealised eta^(2/3) floor, because |f| and
  |f'''| are not 1 and the grid samples two steps per decade. Both the
  README caption and `docs/theory.md` now say which part of the
  prediction the figure supports (the slopes: -0.99 and +2.00 measured
  against -1 and +2) and which part it only brackets.
- The decode speed-up was quoted as "near 25x" with "repeated runs
  anywhere between 24x and 27x". Nine fresh runs on an Apple M2 spanned
  15x to 37x; `docs/figures.py` itself printed 23.5x, 26.5x and 31.0x on
  three consecutive invocations. README now gives the spread, the
  machine and the protocol instead of a two-digit figure the next run
  will contradict.
- The 0.4.0 entry below called 0.0260 nats a "ceiling" that the bigram
  baseline's 0.0287-nat loss exceeded, which reads as a contradiction.
  0.0260 is I(X; X-2 | X-1), the closed-form edge available to a
  second-order predictor; the extra 0.0027 is finite-sample slack on one
  test draw. Corrected there and in the README paragraph that repeated
  it.

## [0.4.0] - 2026-07-27

### Added
- `docs/validation.md` and `examples/validate.py`: eleven claims checked
  against references outside this repository — the definition of a
  derivative, a triple-loop transcription of the attention equation, an
  ablation of the causal mask, memorising one batch, the 1/sqrt(d_k)
  variance argument, the exact Gaussian-CDF GELU, ln V, `var/(var+eps)`,
  Adam's bias-corrected first step, the architecture's parameter count,
  and the entropy rate of a Markov source. Two rows disagree with their
  reference on purpose and say so; nothing was tuned to make them agree.
- `tests/test_properties.py`: randomised invariants rather than fixtures
  — the probability simplex, softmax shift invariance and its Jacobian,
  temperature monotonicity, LayerNorm's affine invariance and the
  orthogonality of its backward, cross-entropy's zero-sum gradient,
  attention outputs inside the convex hull of their values, causal
  masking under intervention, batch-gradient decomposition, gradient
  accumulation, Adam's first step, and cached-vs-full decoding over
  random architectures. Fixed seeds, a few hundred draws each.
- `tests/test_validation.py`: the same ladder on CI-sized budgets, so
  every row of `docs/validation.md` is pinned on every push — including
  the two documented disagreements, which must keep disagreeing.
- `examples/regime_handoff.py`: a worked hand-off from an upstream
  labeller to a downstream consumer of distributions, with the upstream
  source inlined as a second-order Markov chain so the model can be
  graded against a closed-form entropy rate. Held-out cross-entropy
  0.9085 nats/label against an oracle 0.9068 on the same tokens; the
  bigram baseline loses 0.0287 nats to it, of which 0.0260 is the
  closed-form edge I(X; X-2 | X-1) that looking two labels back is worth
  and the rest is finite-sample slack on one 4,000-label draw.

### Changed
- The decode figure counts operations instead of timing them.
  `docs/figures.py` instruments `GPT`, checks the count against the
  closed form, and plots integers — so `docs/kv_cache.png` regenerates
  byte-identically and the README table cannot drift from the figure.
  Wall-clock is printed, with the caveat that it lands in the low tens
  of x rather than the 132x the position counts show, and why.
- `docs/positional_generalization.png` is drawn by `docs/figures.py` in
  the same house style as the other two figures, with the annotation that
  used to collide with the window marker moved and a second panel showing
  what each model actually emits.
- All three figures are now deterministic: fixed seeds and counted
  operations, so they regenerate byte-identically on a given
  matplotlib build. Pixel dimensions still move between matplotlib
  versions; the content does not.

### Fixed
- `ruff check .` is clean under the repo's own configuration. The `E702`
  exemption in `pyproject.toml` existed only to tolerate semicolon-joined
  statements in two finite-difference loops; the loops are rewritten and
  the exemption is gone, so CI lints what it claims to lint.
- README no longer says the key/value-cache test "asserts the consequence
  to 1.1e-15". It asserts agreement below 1e-12; 1.1e-15 is what the test
  measures. Same correction in `docs/theory.md` and in the 0.3.0 entry
  below.
- README no longer implies the exhaustive 1,312-scalar sweep runs in CI.
  It runs by hand; CI checks all 29 tensors with sampled coordinates.
- The hero figure caption had the finite-difference floor at eta^(1/3);
  eta^(1/3) is the optimal *step*, and the floor is eta^(2/3), as
  `docs/theory.md` derives and panel (c) measures.

## [0.3.0] - 2026-07-27

### Added
- Key/value cache for decoding: `MultiHeadAttention.forward_step`,
  `TransformerBlock.forward_step`, `GPT.forward_step`, and
  `GPT.generate(..., use_cache=True)` — now the default. The test
  requires cached and recomputed logits to agree to better than 1e-12 at
  every step, greedy and sampled; the worst it measures is 1.1e-15.
  Decoding 256 tokens from a 4-block model does 132x less work. The cache
  is rebuilt whenever the context window slides past `max_T`: learned
  absolute positions renumber every surviving token and invalidate it.
- `named_params()` on `GPT`, `TransformerBlock`, `MultiHeadAttention` and
  `Linear`. `params()` is now defined as its values, so the two cannot
  drift, and `tests/test_gradcheck.py` is parametrised over it instead of
  over a hand-maintained copy of the model's structure.
- `docs/figures.py`, which regenerates every README figure from a cold
  start: the gradient-verification tearsheet and the decode-cost scaling.
- `docs/theory.md`: why incremental decoding is exact rather than merely
  close, and the truncation-versus-cancellation derivation that fixes the
  finite-difference step at ~1e-5.
- `tests/test_kv_cache.py`: cached-vs-full equivalence, including across
  a window slide, plus `forward_step` input validation.

### Changed
- `examples/gradcheck.py` now checks all 29 parameter tensors (1,312
  scalars) and prints a per-tensor error table; it used to check
  `token_emb` alone, which was the one tensor the tests already covered.
- README rewritten around the gradient evidence rather than a feature
  list.

## [0.2.0] - 2026-07-09

### Fixed
- The flagship demo: `examples/train_pattern.py` trained on a single
  fixed window, so the position embeddings memorised absolute positions
  and greedy generation degenerated to `...,5,4,5,4` once the context
  window slid past the training window. Training windows are now
  sampled at random offsets and generation reproduces the pattern
  exactly (pinned by `tests/test_generalization.py`).
- Cross-entropy now computed as logits - logsumexp; the previous
  `log(softmax(x) + 1e-300)` saturated at ~690.8 once the target
  probability underflowed.
- `GPT.forward` rejects out-of-range token ids with `ValueError`;
  negative ids previously wrapped silently to the wrong embedding rows.
- `GPT.generate`: `temperature=0` is exact greedy decoding, negative
  temperatures raise instead of being clamped, and the blanket
  `np.errstate(all="ignore")` suppression is gone.
- `test_gpt_loss_decreases_on_repeat_task` asserted only that losses
  were not None; replaced with a real learning-threshold test.

### Added
- `tests/test_gradcheck.py`: finite-difference check of all 29
  parameter tensors of a 2-block model (with repeated token ids to
  stress the embedding scatter-add).
- `examples/positional_generalization.py` + docs: fixed-offset vs
  random-offset per-position accuracy study with figure.
- Sampling/validation edge-case tests (`tests/test_generate.py`).

## [0.1.0] - 2026-12-XX

### Added
- `tfs.ops`: softmax, layernorm (+ backward), GELU (+ backward),
  softmax cross-entropy (+ backward), Param wrapper.
- `tfs.attention.MultiHeadAttention`: causal multi-head attention with
  forward + manual backward (softmax-Jacobian, head split/merge).
- `tfs.layers`: `Linear`, `FFN` (GELU), pre-norm `TransformerBlock`.
- `tfs.model.GPT`: decoder-only LM with token + position embeddings,
  stacked blocks, final LayerNorm, LM head.
- `tfs.model.GPT.generate`: autoregressive sampling with temperature.
- `tfs.train.AdamLite`: Adam optimiser.
- End-to-end gradient check against finite differences.
- Example training run on a period-5 pattern (loss < 0.01 in 400 steps).
- CI on Python 3.11 + 3.12.
