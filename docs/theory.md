# Transformer-from-scratch theory

## The whole model in equations

Input ids \(x \in \{1, \ldots, V\}^{T}\). Embed:
\[
h_0 = E_{\text{tok}}[x] + E_{\text{pos}}[0:T].
\]

For each of \(N\) blocks \(\ell\):
\[
\begin{aligned}
\tilde h_\ell &= h_{\ell-1} + \mathrm{Attn}(\mathrm{LN}(h_{\ell-1})) \\
h_\ell &= \tilde h_\ell + \mathrm{FFN}(\mathrm{LN}(\tilde h_\ell))
\end{aligned}
\]

Final logits: \(\mathrm{logits} = W_{\text{LM}} \, \mathrm{LN}(h_N)\).

Loss: \(L = -\frac{1}{NT}\sum \log \mathrm{softmax}(\mathrm{logits})_{\text{target}}\).

## Causal multi-head attention

Per head:
\[
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_{\text{head}}}} + M\right) V,
\]
where \(M\) is the causal mask (upper-triangle \(-\infty\)).

Heads are computed in parallel, then concatenated and projected via \(W_O\).

## Manual backward pass — the bits people get wrong

### Softmax-of-attention backward

If \(p = \mathrm{softmax}(s)\) along the last axis, then
\[
\frac{\partial L}{\partial s} = p \odot \left(\frac{\partial L}{\partial p} - \sum_j \frac{\partial L}{\partial p_j} p_j\right).
\]
The subtraction is the per-row dot-product correction; missing it produces
gradients that are off by exactly the centroid.

### LayerNorm backward

For \(y = \gamma \, \hat x + \beta\) with \(\hat x = (x - \mu)/\sqrt{\sigma^2 + \epsilon}\):
\[
\frac{\partial L}{\partial x} = \frac{1}{N \sqrt{\sigma^2 + \epsilon}}
\left(N\,(\gamma \odot \frac{\partial L}{\partial y}) - \sum_j \gamma_j \, \partial_y L_j - \hat x \, \sum_j \hat x_j (\gamma_j \partial_y L_j) \right).
\]

### Embedding-table backward

Embedding lookup is a discrete index op. The gradient is **scatter-add**:
\[
\frac{\partial L}{\partial E_k} = \sum_{t : x_t = k} \frac{\partial L}{\partial h_t}.
\]
We use `np.add.at(grad, ids, d_x)` to do this safely for repeated indices.

### Cross-entropy without the epsilon fudge

The tempting implementation, `-log(softmax(x)[target] + eps)`, is wrong
in the one regime where the loss value carries information: once the
target probability underflows, the loss silently saturates at
\(-\log \epsilon\) no matter how wrong the model is. The exact identity
\[
\log \mathrm{softmax}(x)_t = x_t - \mathrm{logsumexp}(x)
\]
needs no epsilon, is finite for any float64 logits, and is what
`tfs.ops.softmax_crossentropy` computes (with the usual max-subtraction
inside the logsumexp).

### Why a key/value cache changes nothing

Decoding one token at a time against cached keys and values is usually
sold as an approximation you get away with. It is not an approximation at
all, and the reason is worth writing down.

Let \(x^{(\ell)}_s\) be the input to block \(\ell\) at position \(s\).
Three facts about a decoder-only block:

1. LayerNorm normalises over the **feature** axis, so
   \(\mathrm{LN}(x)_s\) depends only on \(x_s\).
2. The FFN is position-wise: \(\mathrm{FFN}(x)_s\) depends only on \(x_s\).
3. Causal attention at query \(t\) reads only \(K_{\le t}, V_{\le t}\),
   and \(K_s = \mathrm{LN}(x^{(\ell)}_s) W_K\) depends only on position
   \(s\).

By induction over \(\ell\), \(x^{(\ell)}_s\) is a function of tokens
\(\le s\) only. Appending a token at position \(t+1\) therefore cannot
change any \(K_s, V_s\) for \(s \le t\): the cached tensors are the
*same numbers* the full forward pass would recompute, and the only work
left is one query row against them. `tests/test_kv_cache.py` pins the
consequence: it requires the cached and recomputed logits to agree to
better than \(10^{-12}\) at every step, and the worst disagreement it
measures is \(1.1 \times 10^{-15}\) — five units in the last place of a
float64, which is what "the same numbers" looks like once the additions
have happened in a different order.

Two details fall out of the same argument. There is no causal mask in
the cached path (`MultiHeadAttention.forward_step`): every cached key is
already in the past, so there is nothing to mask. And the argument uses
"position \(s\)" as an *absolute* index — which is exactly what a
learned position table encodes. Once the context window slides past
`max_T`, every surviving token gets a new position id, \(x^{(0)}_s\)
changes, and the whole cache is garbage. `GPT.generate` rebuilds it.
Relative position schemes do not pay that cost; this is one of the
quieter reasons they won.

### How big should the finite-difference step be?

The gradient check is only evidence if its own error is smaller than the
error it is looking for. For a central difference,
\[
\frac{f(x+\varepsilon) - f(x-\varepsilon)}{2\varepsilon}
= f'(x) + \frac{\varepsilon^2}{6} f'''(\xi) + O(\varepsilon^4),
\]
so truncation error falls like \(\varepsilon^2\). But each evaluation of
\(f\) carries relative rounding error \(\eta \approx 2.2\times10^{-16}\)
(float64), and the subtraction cancels the leading digits, leaving an
absolute error of about \(\eta |f| / \varepsilon\). The total is
\[
E(\varepsilon) \approx \frac{\varepsilon^2}{6}|f'''| + \frac{\eta |f|}{\varepsilon},
\qquad
\varepsilon^{*} = \left(\frac{3\eta|f|}{|f'''|}\right)^{1/3}.
\]
For a loss with \(|f| \sim |f'''| \sim 1\) that is \(\varepsilon^{*}
\approx 9\times10^{-6}\), with a floor of \(E \sim \eta^{2/3} \approx
4\times10^{-11}\). Panel (c) of the README figure is that curve,
measured, and it is worth being exact about how well the prediction
does. The shape lands: fitting the outer decades of the sweep gives
slopes \(-0.99\) and \(+2.00\) against the predicted \(-1\) and
\(+2\). The position is close but not equal: the best sampled step is
\(\varepsilon = 3.2\times10^{-5}\), a factor \(3.6\) to the right of
\(\varepsilon^{*}\), and the median relative error there is
\(4.0\times10^{-10}\), a factor \(11\) above the \(\eta^{2/3}\)
floor. Neither gap is mysterious — \(|f|\) and \(|f'''|\) are set to 1
in the idealisation and are not 1 for this loss, and the sweep samples
only two steps per decade — consecutive grid points sit a factor
\(\sqrt{10} \approx 3.2\) apart — so "the minimum" means the best grid
point rather than the true one. The grid point nearest
\(\varepsilon^{*}\) is \(10^{-5}\), one step below the measured minimum
at \(3.2\times10^{-5}\); one grid step times the \(1.1\) by which
\(\varepsilon^{*}\) misses the grid is the whole factor \(3.6\). What
the figure supports is the tradeoff and its slopes, not the constants.

The same algebra explains why we never use a forward difference. There
truncation is \(O(\varepsilon)\), so \(\varepsilon^{*} \sim
\sqrt{\eta} \approx 1.5\times10^{-8}\) and the best achievable error is
\(\sqrt{\eta} \approx 10^{-8}\) — three orders of magnitude worse, and
close enough to a real bug's signature to be useless as evidence.

### Position embeddings are lookup tables, not rules

If the training distribution shows only one phase of a periodic
sequence at each absolute position, the cheapest solution is a
position-embedding lookup table — which breaks the moment generation
slides the context window to an unseen phase. We hit exactly this with
the toy task; the failure and the fix (random-offset training windows)
are measured in [positional-generalization.md](positional-generalization.md).

## Training the toy task

We train on a periodic sequence: predict the next character in
`[1,2,3,4,5,1,2,3,4,5,...]`, with training windows sampled at random
offsets so every phase appears at every position. With `d_model=24`,
`n_heads=3`, `n_blocks=2`, Adam at lr=5e-3, loss drops from
\(\log 6 \approx 1.79\) to below \(10^{-3}\) in 400 steps and greedy
decoding reproduces the pattern exactly (enforced by
`tests/test_generalization.py`).

## Every equation above, graded

Nothing on this page is worth much if it only agrees with the code that
implements it. [validation.md](validation.md) grades each of these
against something outside the repository: the softmax Jacobian and the
LayerNorm backward through a finite-difference sweep of all 1,312
parameter scalars, the attention forward against a triple-loop
transcription of the published equation, the causality argument behind
the cache against an ablation that rewrites the future 360 times, and
the GELU against the exact \(x\Phi(x)\) it approximates — which it
misses by \(4.7\times10^{-4}\), on purpose. Two of the eleven rows
there disagree with their reference, and they stay in the table.

## References

- Vaswani et al. (2017), *Attention is all you need* — the architecture.
- Radford et al. (2018), *Improving language understanding by generative
  pre-training* — the decoder-only setup this repo builds.
- Ba, Kiros & Hinton (2016), *Layer normalization*; Xiong et al. (2020),
  *On layer normalization in the transformer architecture* — why the
  blocks here are pre-norm.
- Press, Teukolsky, Vetterling & Flannery, *Numerical Recipes*, §5.7 —
  the \(\varepsilon \sim \eta^{1/3}\) result for central differences.
- Nocedal & Wright, *Numerical Optimization*, §8.1 — the same tradeoff
  from the optimisation side, including why gradient checks are done in
  double precision.
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*
  (and nanoGPT) — the canonical pedagogical treatment, with autograd
  doing the backward pass that this repo writes out by hand.
- Hendrycks & Gimpel (2016), *Gaussian Error Linear Units (GELUs)* — the
  exact activation and the tanh approximation `tfs.ops.gelu` implements.
- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization* —
  Algorithm 1, including the bias correction `AdamLite` applies.
