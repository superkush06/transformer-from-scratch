# transformer-from-scratch

[![ci](https://github.com/superkush06/transformer-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/superkush06/transformer-from-scratch/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A decoder-only transformer (GPT-style) in pure NumPy. **Manual backprop
> for every op** — embeddings (scatter-add), multi-head causal attention
> (softmax-Jacobian by hand), LayerNorm (mean/var-aware), FFN with GELU,
> and softmax cross-entropy. No PyTorch / JAX / autograd library at all.

## TL;DR

```python
from tfs.model import GPT
from tfs.train import AdamLite

model = GPT(vocab_size=64, d_model=64, n_heads=4, d_ff=128,
            n_blocks=2, max_T=64, seed=0)
opt = AdamLite(model.params(), lr=3e-3)

for step in range(epochs):
    for p in model.params(): p.zero_grad()
    loss = model.loss_and_grads(ids, targets)
    opt.step()
```

## What's inside

- `tfs/ops.py` — softmax, LayerNorm, GELU, cross-entropy with their
  hand-derived backward passes.
- `tfs/attention.py` — causal multi-head attention with manual backward.
  Gradients verified against finite differences in tests.
- `tfs/layers.py` — Linear, FFN, pre-norm TransformerBlock.
- `tfs/model.py` — `GPT` class with `forward`, `loss_and_grads`,
  `generate(prompt, max_new, temperature)`.
- `tfs/train.py` — `AdamLite` optimiser.

## Example output

`PYTHONPATH=. python3 examples/train_pattern.py` on a period-5 sequence:

```
step    loss
    0  1.7918
   40  0.8421
   80  0.3104
  120  0.1132
  160  0.0498
  ...
  399  0.0079

prompt=[1, 2, 3]
generation=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]
target    =[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]
```

`examples/gradcheck.py` verifies the end-to-end backprop:

```
token_emb grad max abs error: 1.84e-06
token_emb grad max rel error: 6.41e-07
PASS: rel < 1e-3
```

## Theory primer

See [`docs/theory.md`](docs/theory.md) for the equations and the *exact*
backward formulas (softmax-jacobian correction, LayerNorm gradient
decomposition, embedding scatter-add).

## Install

```bash
git clone https://github.com/superkush06/transformer-from-scratch.git
cd transformer-from-scratch
pip install -e ".[dev]"
pytest
```

## Roadmap

- [ ] Mixed-precision (numpy doesn't help, so a JAX backend).
- [ ] Flash-attention-style memory-efficient implementation.
- [ ] Tokenizers: BPE on tiny corpora.
- [ ] Train on TinyStories with a GPT2-small budget.

## License

MIT — see [LICENSE](LICENSE).
