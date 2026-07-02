# Changelog

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
