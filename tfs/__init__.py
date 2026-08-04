"""tfs — transformer from scratch (numpy + manual backprop)."""

from .attention import MultiHeadAttention
from .layers import FFN, Linear, TransformerBlock
from .model import GPT
from .ops import (
    Param,
    gelu,
    gelu_backward,
    layernorm,
    layernorm_backward,
    softmax,
    softmax_crossentropy,
)
from .train import AdamLite

__version__ = "0.4.2"
__all__ = [
    "GPT", "MultiHeadAttention", "TransformerBlock", "Linear", "FFN",
    "AdamLite",
    "Param", "softmax", "softmax_crossentropy",
    "layernorm", "layernorm_backward", "gelu", "gelu_backward",
    "__version__",
]
