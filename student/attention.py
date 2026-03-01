import torch
import math
from einops import einsum
import torch.cuda.nvtx as nvtx

from a1_basics.nn_utils import softmax


@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention with NVTX annotations for profiling.
    This replaces the original function to enable detailed profiling of attention components.
    """
    d_k = K.shape[-1]

    with nvtx.range("attention_matmul_qk"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        with nvtx.range("attention_mask"):
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("attention_softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("attention_matmul_v"):
        output = einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")

    return output


def enable_attention_profiling():
    """Monkey-patch the attention function with NVTX-annotated version."""
    import a1_basics.model as model_module

    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention
