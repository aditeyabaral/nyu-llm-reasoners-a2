"""
FlashAttention-2 Triton implementation.

This module provides an autograd Function that uses Triton for the forward pass.
"""

import math
import torch
import triton
import triton.language as tl
from triton import cdiv

# Import backward function
try:
    from student.flash_attention import flash_attention_backward_compiled
except ImportError:
    from flash_attention import flash_attention_backward_compiled


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    N_K_TILES: tl.constexpr,
):
    """
    FlashAttention-2 forward kernel.
    Each program instance processes one query tile for one batch element.
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Compute base pointers for this batch
    Q_batch_ptr = Q_ptr + batch_index * stride_qb
    K_batch_ptr = K_ptr + batch_index * stride_kb
    V_batch_ptr = V_ptr + batch_index * stride_vb
    O_batch_ptr = O_ptr + batch_index * stride_ob
    L_batch_ptr = L_ptr + batch_index * stride_lb

    # Query tile start
    q_start = query_tile_index * Q_TILE_SIZE

    # Offsets for loading Q tile
    q_offs = q_start + tl.arange(0, Q_TILE_SIZE)
    d_offs = tl.arange(0, D)

    # Load Q tile: (Q_TILE_SIZE, D)
    q_ptrs = Q_batch_ptr + q_offs[:, None] * stride_qq + d_offs[None, :] * stride_qd
    q_mask = (q_offs[:, None] < N_QUERIES) & (d_offs[None, :] < D)
    Qi = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Initialize accumulators
    mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # Loop over key tiles
    for j in range(N_K_TILES):
        k_start = j * K_TILE_SIZE
        k_offs = k_start + tl.arange(0, K_TILE_SIZE)

        # Load K tile: (K_TILE_SIZE, D)
        k_ptrs = K_batch_ptr + k_offs[:, None] * stride_kk + d_offs[None, :] * stride_kd
        k_mask = (k_offs[:, None] < N_KEYS) & (d_offs[None, :] < D)
        Kj = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load V tile: (K_TILE_SIZE, D)
        v_ptrs = V_batch_ptr + k_offs[:, None] * stride_vk + d_offs[None, :] * stride_vd
        Vj = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Compute S = Q @ K^T * scale: (Q_TILE_SIZE, K_TILE_SIZE)
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        # Apply causal mask if needed
        if is_causal:
            causal_mask = q_offs[:, None] >= k_offs[None, :]
            Sij = tl.where(causal_mask, Sij, -1e6)

        # Online softmax
        mij = tl.max(Sij, axis=1)
        mi_new = tl.maximum(mi, mij)
        Pij = tl.exp(Sij - mi_new[:, None])
        li_new = tl.exp(mi - mi_new) * li + tl.sum(Pij, axis=1)
        Oi = tl.exp(mi - mi_new)[:, None] * Oi + tl.dot(Pij, Vj)

        mi = mi_new
        li = li_new

    # Final normalization
    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)

    # Store O: (Q_TILE_SIZE, D)
    o_ptrs = O_batch_ptr + q_offs[:, None] * stride_oq + d_offs[None, :] * stride_od
    o_mask = (q_offs[:, None] < N_QUERIES) & (d_offs[None, :] < D)
    tl.store(o_ptrs, Oi, mask=o_mask)

    # Store L: (Q_TILE_SIZE,)
    l_ptrs = L_batch_ptr + q_offs * stride_lq
    l_mask = q_offs < N_QUERIES
    tl.store(l_ptrs, Li, mask=l_mask)


class FlashAttentionTriton(torch.autograd.Function):
    """
    FlashAttention-2 autograd Function using Triton kernel.
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        assert Q.is_cuda, "Q must be on CUDA"
        assert K.is_cuda, "K must be on CUDA"
        assert V.is_cuda, "V must be on CUDA"

        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        O = torch.empty_like(Q)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=Q.dtype)

        # Use tile sizes that are powers of 2
        Q_TILE_SIZE = min(64, triton.next_power_of_2(n_queries))
        K_TILE_SIZE = min(64, triton.next_power_of_2(n_keys))
        Q_TILE_SIZE = max(16, Q_TILE_SIZE)
        K_TILE_SIZE = max(16, K_TILE_SIZE)

        scale = 1.0 / math.sqrt(d)

        n_q_tiles = cdiv(n_queries, Q_TILE_SIZE)
        n_k_tiles = cdiv(n_keys, K_TILE_SIZE)
        grid = (n_q_tiles, batch_size)

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=scale,
            is_causal=is_causal,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            N_K_TILES=n_k_tiles,
            num_warps=4,
            num_stages=1,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_attention_backward_compiled(Q, K, V, O, dO, L, is_causal)
        return dQ, dK, dV, None
