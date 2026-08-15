# Validation

A passing test suite proves a library agrees with itself. This page is
about whether it agrees with anything else.

Eleven claims, each checked against something that is not this codebase:
the definition of a derivative, an implementation written from the paper
instead of from `tfs/`, a closed-form probability, an information-theoretic
floor. Every number below is printed by one command —

```bash
PYTHONPATH=. python3 examples/validate.py        # ~20 s
```

— and `tests/test_validation.py` runs the same checks on smaller budgets
so CI fails the day one of them stops being true.

Two rows disagree with their reference. They are marked, and the reason is
given; **the numbers here were not tuned to make anything agree.**

## The table

| # | claim | our value | reference value | source |
| - | ----- | --------- | --------------- | ------ |
| 1 | max relative error, hand-derived gradient vs central difference | `3.06e-07` | 0 exactly; float64 floor near η^(2/3) ≈ 4e-11 | the definition of the derivative; Press et al., *Numerical Recipes* §5.7 for the achievable floor |
| 2 | vectorised attention vs a triple-loop implementation of the published equation | `1.67e-16` max abs | 0 — same function | Vaswani et al. (2017), §3.2.1, scaled dot-product attention |
| 3 | largest change to a logit at position ≤ t when every token after t is rewritten | `0.0` | 0 exactly | the causal-mask definition (−1e9 underflows to exactly zero weight in float64) |
| 4 | loss / argmax accuracy after memorising one batch | `2.67e-06` / `1.000` | 0 / 1.000 (deterministic task; start is ln 16 = 2.773) | Karpathy (2019), *A Recipe for Training Neural Networks* — "overfit a single batch" |
| 5 | variance of an unscaled dot product of two unit-variance vectors in 64 dimensions | `64.09` | d_k = 64 | Vaswani et al. (2017), §3.2.1 — the argument for the 1/√d_k factor |
| 6 | `tfs.ops.gelu` vs the exact Gaussian-CDF GELU on [−6, 6] | `4.73e-04` at x = +2.70 | 0, if the exact form were implemented | Hendrycks & Gimpel (2016), *Gaussian Error Linear Units* — both forms **← disagrees** |
| 7 | cross-entropy of a uniform distribution over 17 classes | `2.833213344056216` | ln 17 = `2.833213344056216` | closed form: −ln(1/V) |
| 8 | mean output variance of LayerNorm, γ=1, β=0 | `0.9999972221` | 1.0 — the usual "zero mean, unit variance" | Ba, Kiros & Hinton (2016), *Layer Normalization* **← disagrees** |
| 9 | first Adam step, gradients spanning 12 decades | max abs error `2.04e-16` | −lr·g/(‖g‖+ε), i.e. −lr·sign(g) as ε→0 | Kingma & Ba (2015), *Adam*, Algorithm 1 |
| 10 | parameter count of the README's demo model | `10,080` | `10,080` | closed form: VD + T_max·D + 2D + L(4D² + 2D·d_ff + d_ff + 5D) + DV |
| 11 | held-out cross-entropy on a second-order Markov source | `0.9085` nats/label | `0.9068` — the source scoring the same tokens; `0.9303` in the limit | Cover & Thomas, *Elements of Information Theory* (2nd ed.), §4.2 — entropy rate of a stationary Markov chain |

## Where we do not match, and why

### Row 6 — the GELU is the approximation, and it is off by 4.7e-04

`tfs.ops.gelu` implements

    0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))

which is the tanh approximation from Hendrycks & Gimpel, not the exact
`x·Φ(x)`. Over [−6, 6] the two differ by at most **4.73e-04**, at
x = +2.70. That is five orders of magnitude larger than anything else on
this page, and it is not a bug: GPT-2 uses the same approximation, and the
backward pass in `tfs/ops.py` is the exact derivative *of the tanh form*,
so the layer is internally consistent — which is what the gradient check
in row 1 confirms.

If you want the exact activation, swap both `gelu` and `gelu_backward`
together. Swapping one is the mistake this row exists to make visible.

### Row 8 — LayerNorm does not produce unit variance, and cannot

The layer divides by `sqrt(var + eps)`, so the output variance is
`var / (var + eps)`, not 1. With `eps = 1e-5` and unit-ish inputs that is
**0.9999972221** against a claimed 1.0 — a relative shortfall of 2.8e-06.
The measured value agrees with the closed form `var/(var+eps)` to 2e-16,
so the gap is entirely the variance floor and nothing else.

The reason to state it rather than round it away: the floor is also why
LayerNorm's shift/scale invariance is only approximate, and
`tests/test_properties.py` has to pass `eps=0.0` to assert that invariance
exactly. A page that claimed unit variance would be quietly contradicted by
its own test suite.

### Row 1 — 3e-07 is the floor, not the accuracy

The gradients are not "accurate to 3e-07"; the *measurement* is. A central
difference carries truncation error ~ε²·|f‴|/6 and cancellation error
~η·|f|/ε, and their sum bottoms out around η^(2/3) ≈ 4e-11 for a
well-scaled loss. The worst tensor here sits at 3.06e-07 because its
gradient entries are small relative to the loss, so the relative error is
divided by a small number. `docs/theory.md` derives the tradeoff, and panel
(c) of the README figure measures the V-shaped curve it predicts.

### Row 4 — 2.67e-06 is not zero

Memorising one batch is an optimisation problem with a minimum at
`-inf` logit margins, so the loss approaches zero without reaching it. What
*is* exact is the argmax: all 16 targets are the top-scoring token. Running
6,000 steps instead of 4,000 takes the loss to 7.7e-07 and changes nothing
else, which is the sign of a well-conditioned optimisation rather than a
stuck one.

### Row 11 — the model scores *below* the population entropy rate

The generating chain's entropy rate is 0.9303 nats/label, and the model
scores 0.9085 on held-out data. That looks impossible, and it is worth
being precise about why it is not: 0.9303 is a population average, while
0.9085 is measured on one 4,000-label draw. The oracle — the true chain
scoring those same tokens — gets 0.9068, so this particular test sequence
is simply a slightly easy one, and **0.9068 is the number the model should
be compared against.** It comes in 0.0017 nats above it.

The comparison that carries the actual information is the paired one:

```
  uniform over 4 labels                  1.3863
  order-1 entropy H(X|X-1)   [closed]    0.9563
  bigram MLE, fitted + scored            0.9372
  this model                             0.9085
  oracle: the source itself              0.9068
  entropy rate H(X|X-1,X-2)  [closed]    0.9303
```

The bigram baseline loses 0.0287 nats to the model. The closed-form edge
available to anything that looks two labels back is
I(X; X₋₂ | X₋₁) = **0.0260** nats. The model captures it and a little of the
finite-sample slack besides — which is the strongest statement in this
document, because the reference is a number that exists whether or not this
code does.

Total variation between the model's predictive distribution and the true
conditional, over all 16 two-label contexts: max 0.0864, mean 0.0489,
stationary-weighted mean 0.0351. The worst context carries 0.6% of the
stationary mass — the model is least accurate exactly where the source
almost never goes, which is the expected shape of the error and not a
defect.

## What is *not* validated here

- **No comparison against PyTorch or JAX in this document.** They are not
  dependencies and will not become ones, so every reference on this page is
  either a closed form or an implementation written inside this repository
  from the published equation. The torch comparison lives one layer up:
  [`notebooks/audit.ipynb`](../notebooks/audit.ipynb) rebuilds the same
  architecture with `torch` ops, copies these weights across and diffs the
  gradients — they agree to about `1e-15` in float64 at every architecture
  tried, from 1,312 to 17,536 parameters — and CI's `parity` job runs that
  comparison on every push (`tests/test_torch_parity.py` sweeps five
  architectures; `tests/test_notebook.py` holds the notebook's prose to the
  precision it claims), installing torch from the `[audit]` extra so the
  package itself stays NumPy-only.
- **No claim about numerical behaviour in float32.** Everything runs in
  float64; the gradient check in particular is meaningless at lower
  precision (see `docs/theory.md` on why ε ≈ 1e-5 needs double).
- **No claim about scale.** The largest model here has 0.9M parameters.
  Nothing on this page says anything about what happens at 10⁹.
- **No claim about training dynamics.** There is no schedule, no clipping,
  no weight decay; row 4 shows the optimiser can memorise, not that it
  generalises. Row 11 is the only generalisation claim, and it is on a
  source simple enough to have an answer.

## Full output

```
validating tfs against references outside it

[1] gradients: 1,312 scalars over 29 tensors, 1,312 signs agree, worst relative error 3.06e-07
[2] attention: max |vectorised - naive| = 1.67e-16 over a (2, 6, 8) input; outputs are O(0.93)
[3] causal mask: 360 interventions on the future, largest change to any earlier logit = 0.0
[4] overfit: loss 2.9173 -> 2.67e-06 in 4000 steps (ln 16 = 2.7726); argmax accuracy 1.000
[5] score scale: Var(q·k) = 64.09 for d_k = 64; after dividing by sqrt(d_k), Var = 1.0014
[6] GELU: max |tanh form - x*Phi(x)| = 4.73e-04 at x = +2.699  (this is a real disagreement)
[7] uniform cross-entropy: 2.833213344056216 vs ln 17 = 2.833213344056216
[8] LayerNorm: mean output variance 0.9999972221; var/(var+eps) predicts 0.9999972221; the textbook claim is 1
[9] Adam: max |step_1 - (-lr*g/(|g|+eps))| = 2.04e-16; against the eps -> 0 form -lr*sign(g) it is 4.90e-08
[10] parameters: 10,080 counted, 10,080 from the architecture (2 x 4,776 in blocks)
[11] entropy rate: model 0.9085, oracle 0.9068, bigram 0.9372 nats/label; closed-form rate 0.9303, H(X|X-1) 0.9563

==========================================================================================
[ 1] ok   max relative error, hand-derived gradient vs central difference
          ours       3.06e-07
          reference  0 (exact), with a float64 floor near eta^(2/3) ~ 4e-11
          source     the definition of the derivative; Press et al., Numerical Recipes §5.7 for
                     the achievable floor
          note       1,312 scalars, all 29 parameter tensors, eps=1e-5
------------------------------------------------------------------------------------------
[ 2] ok   max absolute difference, vectorised attention vs a loop implementation of
          the paper's equation
          ours       1.67e-16
          reference  0 (same function)
          source     Vaswani et al. (2017), §3.2.1, scaled dot-product attention
          note       2 heads, T=6; the reference shares no code with tfs/
------------------------------------------------------------------------------------------
[ 3] ok   largest change to a logit at position <= t when every token after t is
          rewritten
          ours       0.0
          reference  0 exactly
          source     the causal-mask definition; -1e9 underflows to exactly zero weight in
                     float64
          note       360 interventions (resample and permute), T=9
------------------------------------------------------------------------------------------
[ 4] ok   final loss and argmax accuracy after memorising one batch
          ours       2.67e-06 / 1.000
          reference  0 / 1.000 (the task is deterministic; start is ln 16 = 2.773)
          source     Karpathy (2019), A Recipe for Training Neural Networks — 'overfit a single
                     batch'
          note       1x16 sequence, random targets, 4000 Adam steps at lr 1e-2
------------------------------------------------------------------------------------------
[ 5] ok   variance of an unscaled dot product of two unit-variance vectors in 64
          dimensions
          ours       64.09
          reference  d_k = 64
          source     Vaswani et al. (2017), §3.2.1 — the argument for the 1/sqrt(d_k) factor
          note       200,000 Monte-Carlo draws; scaled variance 1.0014
------------------------------------------------------------------------------------------
[ 6] DIFF max absolute difference between tfs.ops.gelu and the exact Gaussian-CDF
          GELU on [-6, 6]
          ours       4.73e-04 at x = +2.70
          reference  0 if the exact form were implemented
          source     Hendrycks & Gimpel (2016), Gaussian Error Linear Units — both the exact
                     x*Phi(x) and the tanh approximation
          note       deliberate: the repo ships the tanh approximation, as GPT-2 does, and its
                     derivative is the one written out in tfs/ops.py
------------------------------------------------------------------------------------------
[ 7] ok   cross-entropy of a uniform distribution over 17 classes
          ours       2.833213344056216
          reference  ln 17 = 2.833213344056216
          source     closed form: -ln(1/V)
          note       exact to the last bit of float64
------------------------------------------------------------------------------------------
[ 8] DIFF mean output variance of LayerNorm with gamma=1, beta=0
          ours       0.9999972221
          reference  1.0 exactly — the usual 'zero mean, unit variance' claim
          source     Ba, Kiros & Hinton (2016), Layer Normalization
          note       matches var/(var+eps) = 0.9999972221 to 2e-16, so the gap is the variance
                     floor and nothing else
------------------------------------------------------------------------------------------
[ 9] ok   first Adam step, for gradients spanning 12 orders of magnitude
          ours       |error| <= 2.04e-16
          reference  -lr*g/(|g| + eps), i.e. -lr*sign(g) = 1e-03 as eps -> 0
          source     Kingma & Ba (2015), Adam, Algorithm 1 — after bias correction m1_hat = g
                     and v1_hat = g^2
          note       the magnitude of the gradient cancels; the 5e-08 residual against plain
                     sign(g) is exactly lr*eps/(|g|+eps)
------------------------------------------------------------------------------------------
[10] ok   parameter count of the README's demo model
          ours       10,080
          reference  10,080
          source     closed form: VD + T_max·D + 2D + L(4D² + 2D·d_ff + d_ff + 5D) + DV
          note       V=6, D=24, heads=3, d_ff=48, blocks=2, max_T=8
------------------------------------------------------------------------------------------
[11] ok   held-out cross-entropy on a second-order Markov source, against the Bayes
          floor
          ours       0.9085 nats/label
          reference  0.9068 (the source scoring the same tokens); H = 0.9303 in the limit
          source     Cover & Thomas, Elements of Information Theory (2nd ed.), §4.2 — the
                     entropy rate of a stationary Markov chain
          note       the order-1 baseline scores 0.9372; its handicap I(X;X-2|X-1) = 0.0260
                     nats, and it loses 0.0287
------------------------------------------------------------------------------------------
9/11 rungs agree with their reference. Row(s) 6, 8 disagree on purpose; docs/validation.md says why.
```

## References

- Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin
  (2017), *Attention Is All You Need* — §3.2.1 for scaled dot-product
  attention and the argument for the 1/√d_k factor.
- Hendrycks & Gimpel (2016), *Gaussian Error Linear Units (GELUs)* — the
  exact `x·Φ(x)` and the tanh approximation this repo implements.
- Ba, Kiros & Hinton (2016), *Layer Normalization*.
- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization* —
  Algorithm 1, including the bias-correction terms row 9 depends on.
- Karpathy (2019), *A Recipe for Training Neural Networks* — the
  "overfit a single batch" step that row 4 implements.
- Press, Teukolsky, Vetterling & Flannery, *Numerical Recipes*, §5.7 — the
  optimal finite-difference step for a central difference.
- Cover & Thomas, *Elements of Information Theory* (2nd ed.), §4.2 — the
  entropy rate of a stationary Markov chain, which row 11 uses as a floor.
