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

## Training the toy task

We train on a periodic sequence: predict the next character in
`[1,2,3,4,5,1,2,3,4,5,...]`. With `d_model=24`, `n_heads=3`, `n_blocks=2`,
Adam at lr=5e-3, loss drops from \(\log 6 \approx 1.79\) to below \(0.1\)
in ~400 steps.

## References

- Vaswani et al. (2017), *Attention is all you need*.
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*
  (YouTube + nanoGPT) — the canonical pedagogical resource.
- The original GPT paper (Radford et al. 2018) for the decoder-only setup.
