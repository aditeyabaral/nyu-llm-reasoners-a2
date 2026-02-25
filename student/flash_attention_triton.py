"""
FlashAttention-2 Triton kernel implementation.
"""

import math
import torch
import triton
import triton.language as tl
from triton import cdiv

from student.flash_attention import flash_attention_backward_compiled


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
):
    """
    FlashAttention-2 forward kernel.

    Each program instance processes one query tile for one batch element.
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q tile (stays constant for this program instance)
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)

    # Initialize running values (float32 for numerical stability)
    mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  # Running max
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # Running sum
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # Running output

    # Number of key tiles
    n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # Query indices for causal masking
    q_start = query_tile_index * Q_TILE_SIZE
    q_indices = q_start + tl.arange(0, Q_TILE_SIZE)  # (Q_TILE_SIZE,)

    # Loop over key tiles
    for j in range(n_k_tiles):
        k_start = j * K_TILE_SIZE

        # Load K and V tiles
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)

        # Compute attention scores: Sij = Qi @ Kj^T * scale
        # (Q_TILE_SIZE, D) @ (D, K_TILE_SIZE) -> (Q_TILE_SIZE, K_TILE_SIZE)
        Sij = tl.dot(Qi.to(tl.float32), tl.trans(Kj.to(tl.float32))) * scale

        # Apply causal mask if needed
        if is_causal:
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)  # (K_TILE_SIZE,)
            # Mask where query_idx < key_idx (future tokens)
            causal_mask = q_indices[:, None] >= k_indices[None, :]  # (Q_TILE_SIZE, K_TILE_SIZE)
            Sij = tl.where(causal_mask, Sij, -1e6)

        # Online softmax update
        # m_ij = rowmax(Sij)
        mij = tl.max(Sij, axis=1)  # (Q_TILE_SIZE,)

        # m_new = max(m_old, m_ij)
        mi_new = tl.maximum(mi, mij)

        # P_tilde = exp(Sij - m_new)
        Pij_tilde = tl.exp(Sij - mi_new[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

        # l_new = exp(m_old - m_new) * l_old + rowsum(P_tilde)
        li_new = tl.exp(mi - mi_new) * li + tl.sum(Pij_tilde, axis=1)

        # O_new = exp(m_old - m_new) * O_old + P_tilde @ Vj
        Oi = tl.exp(mi - mi_new)[:, None] * Oi + tl.dot(Pij_tilde.to(Vj.dtype), Vj.to(tl.float32))

        # Update running values
        mi = mi_new
        li = li_new

        # Advance K and V block pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Final scaling: O = O / l
    Oi = Oi / li[:, None]

    # Compute logsumexp: L = m + log(l)
    Li = mi + tl.log(li)

    # Store outputs (cast back to input dtype)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li.to(L_block_ptr.type.element_ty), boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    """
    FlashAttention-2 autograd Function using Triton kernel.

    Forward pass uses Triton kernel.
    Backward pass uses recomputation with torch.compile.
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """
        Forward pass of FlashAttention-2 using Triton.

        Args:
            ctx: Autograd context
            Q: Query tensor (batch, n_queries, d)
            K: Key tensor (batch, n_keys, d)
            V: Value tensor (batch, n_keys, d)
            is_causal: Whether to apply causal masking

        Returns:
            O: Output tensor (batch, n_queries, d)
        """
        assert Q.is_cuda, "Q must be on CUDA"
        assert K.is_cuda, "K must be on CUDA"
        assert V.is_cuda, "V must be on CUDA"

        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        # Ensure contiguous
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        # Allocate outputs
        O = torch.empty_like(Q)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=Q.dtype)

        # Tile sizes (must be powers of 2 and at least 16)
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        # Adjust tile sizes for small dimensions
        if n_queries < Q_TILE_SIZE:
            Q_TILE_SIZE = triton.next_power_of_2(n_queries)
        if n_keys < K_TILE_SIZE:
            K_TILE_SIZE = triton.next_power_of_2(n_keys)

        # Ensure minimum tile size of 16
        Q_TILE_SIZE = max(16, Q_TILE_SIZE)
        K_TILE_SIZE = max(16, K_TILE_SIZE)

        scale = 1.0 / math.sqrt(d)

        # Launch grid: (num_query_tiles, batch_size)
        n_q_tiles = cdiv(n_queries, Q_TILE_SIZE)
        grid = (n_q_tiles, batch_size)

        # Launch kernel
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
        )

        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of FlashAttention-2 using recomputation.
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_attention_backward_compiled(Q, K, V, O, dO, L, is_causal)
        return dQ, dK, dV, None
