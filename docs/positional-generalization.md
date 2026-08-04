# Case study: position embeddings memorise absolute position

The original demo trained the period-5 model on **one fixed window** —
`pattern[:8]` at every step. Training loss fell to 0.0005, the model
looked perfect, and generation still broke:

```
fixed-offset  greedy: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 4, 5, 4]
random-offset greedy: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]
```

(Both lines are real output from `examples/positional_generalization.py`.)

## What actually happened

With a single training window, the cheapest way to zero the loss is a
**position -> token lookup table**: the position embeddings alone
determine the answer ("position 3 is always followed by 4"), and the
attention heads never need to learn the period-5 *algorithm*.

That table is only valid for one phase alignment. `generate` keeps the
last `max_T = 8` tokens as context, so from position 9 onward the
window has slid: the model now sees `[3,4,5,1,2,3,4,5]` at positions it
only ever saw `[1,2,3,4,5,1,2,3]`. The lookup table answers for the
wrong phase — greedy decoding locks onto `...4,5,4,5` and never
recovers.

## Measuring it

`examples/positional_generalization.py` trains the model both ways and
teacher-forces the true sequence through each, recording greedy
next-token accuracy at every position:

```
 positions  fixed-offset  random-offset
       3-8          1.00           1.00
      9-39          0.42           1.00
```

Predicting position `t` uses window `seq[t-8:t]`; `t = 9` is the first
window the fixed-offset model has never seen, and that is exactly where
its curve collapses:

![per-position accuracy](positional_generalization.png)

The fixed-offset curve oscillating between 0 and 1 after position 8 is
the phase error drifting in and out of alignment with the true
sequence — not partial knowledge.

## The fix, and the regression test

`examples/train_pattern.py` now samples training windows at random
offsets into the sequence (`sample_windows`), so every phase of the
period appears at every position and the position -> token shortcut
stops being available. The model is forced to learn the relative rule
("copy the token from 5 steps back"), which survives the sliding
window.

`tests/test_generalization.py` pins this: greedy generation must
reproduce the period-5 pattern **exactly** for 25 tokens — 20 tokens
past the training window, by which point the context has slid through
every phase.

## The general lesson

Learned absolute position embeddings interpolate nothing. If training
never shows a phase/offset at a given position, the model has no reason
to behave correctly there — a toy-scale version of why GPT-style models
degrade beyond their trained context length, and why relative schemes
(ALiBi, RoPE) exist.
