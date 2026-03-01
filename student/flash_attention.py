"""
FlashAttention-2 implementation in PyTorch.
"""

import math
import torch
from torch import Tensor


def flash_attention_forward_tiled(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    is_causal: bool = False,
    q_tile_size: int = 32,
    k_tile_size: int = 32,
) -> tuple[Tensor, Tensor]:
    """
    Tiled FlashAttention-2 forward pass in PyTorch.

    Implements Algorithm 1 from FlashAttention-2 paper using online softmax.

    Args:
        Q: Query tensor of shape (batch, n_queries, d)
        K: Key tensor of shape (batch, n_keys, d)
        V: Value tensor of shape (batch, n_keys, d)
        is_causal: Whether to apply causal masking
        q_tile_size: Tile size for queries (Bq)
        k_tile_size: Tile size for keys (Bk)

    Returns:
        O: Output tensor of shape (batch, n_queries, d)
        L: Logsumexp tensor of shape (batch, n_queries)
    """
    batch_size, n_queries, d = Q.shape
    _, n_keys, _ = K.shape

    scale = 1.0 / math.sqrt(d)

    # Number of tiles
    n_q_tiles = math.ceil(n_queries / q_tile_size)
    n_k_tiles = math.ceil(n_keys / k_tile_size)

    # Initialize output
    O = torch.zeros_like(Q)
    # L stores logsumexp for each query position
    L = torch.zeros(batch_size, n_queries, device=Q.device, dtype=Q.dtype)

    # Iterate over query tiles
    for i in range(n_q_tiles):
        q_start = i * q_tile_size
        q_end = min((i + 1) * q_tile_size, n_queries)
        Qi = Q[:, q_start:q_end, :]  # (batch, Bq, d)

        # Running values for this query tile (online softmax)
        mi = torch.full((batch_size, q_end - q_start), float("-inf"), device=Q.device, dtype=torch.float32)
        li = torch.zeros(batch_size, q_end - q_start, device=Q.device, dtype=torch.float32)
        Oi = torch.zeros(batch_size, q_end - q_start, d, device=Q.device, dtype=torch.float32)

        # Iterate over key tiles
        for j in range(n_k_tiles):
            k_start = j * k_tile_size
            k_end = min((j + 1) * k_tile_size, n_keys)
            Kj = K[:, k_start:k_end, :]  # (batch, Bk, d)
            Vj = V[:, k_start:k_end, :]  # (batch, Bk, d)

            # Compute attention scores for this tile: Sij = Qi @ Kj^T * scale
            # (batch, Bq, d) @ (batch, d, Bk) -> (batch, Bq, Bk)
            Sij = torch.bmm(Qi.float(), Kj.float().transpose(-2, -1)) * scale

            # Apply causal mask if needed
            if is_causal:
                # Create mask: query_idx >= key_idx
                q_indices = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_indices = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                causal_mask = q_indices >= k_indices  # (Bq, Bk)
                Sij = torch.where(
                    causal_mask.unsqueeze(0), Sij, torch.tensor(-1e6, device=Q.device, dtype=torch.float32)
                )

            # Online softmax update
            # m_ij = rowmax(Sij)
            mij = Sij.max(dim=-1).values  # (batch, Bq)

            # m_new = max(m_old, m_ij)
            mi_new = torch.maximum(mi, mij)

            # P_tilde = exp(Sij - m_new)
            Pij_tilde = torch.exp(Sij - mi_new.unsqueeze(-1))  # (batch, Bq, Bk)

            # l_new = exp(m_old - m_new) * l_old + rowsum(P_tilde)
            li_new = torch.exp(mi - mi_new) * li + Pij_tilde.sum(dim=-1)

            # O_new = exp(m_old - m_new) * O_old + P_tilde @ Vj
            Oi = torch.exp(mi - mi_new).unsqueeze(-1) * Oi + torch.bmm(Pij_tilde, Vj.float())

            # Update running values
            mi = mi_new
            li = li_new

        # Final scaling: O = O / l
        Oi = Oi / li.unsqueeze(-1)

        # Compute logsumexp: L = m + log(l)
        Li = mi + torch.log(li)

        # Store results
        O[:, q_start:q_end, :] = Oi.to(Q.dtype)
        L[:, q_start:q_end] = Li.to(Q.dtype)

    return O, L


@torch.compile
def flash_attention_backward_compiled(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    O: Tensor,
    dO: Tensor,
    L: Tensor,
    is_causal: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    FlashAttention-2 backward pass using recomputation.

    Args:
        Q: Query tensor (batch, n_queries, d)
        K: Key tensor (batch, n_keys, d)
        V: Value tensor (batch, n_keys, d)
        O: Output from forward pass (batch, n_queries, d)
        dO: Gradient of output (batch, n_queries, d)
        L: Logsumexp from forward pass (batch, n_queries)
        is_causal: Whether causal masking was applied

    Returns:
        dQ, dK, dV: Gradients for Q, K, V
    """
    batch_size, n_queries, d = Q.shape
    _, n_keys, _ = K.shape
    scale = 1.0 / math.sqrt(d)

    # Compute D = rowsum(dO * O) - Eq. precomputation
    D = (dO * O).sum(dim=-1)  # (batch, n_queries)

    # Recompute S = Q @ K^T / sqrt(d) - Eq. 13
    S = torch.bmm(Q, K.transpose(-2, -1)) * scale  # (batch, n_queries, n_keys)

    # Apply causal mask if needed
    if is_causal:
        q_indices = torch.arange(n_queries, device=Q.device).unsqueeze(1)
        k_indices = torch.arange(n_keys, device=Q.device).unsqueeze(0)
        causal_mask = q_indices >= k_indices
        S = torch.where(causal_mask.unsqueeze(0), S, torch.tensor(-1e6, device=Q.device, dtype=S.dtype))

    # Recompute P = exp(S - L) - Eq. 14
    P = torch.exp(S - L.unsqueeze(-1))  # (batch, n_queries, n_keys)

    # dV = P^T @ dO - Eq. 15
    dV = torch.bmm(P.transpose(-2, -1), dO)  # (batch, n_keys, d)

    # dP = dO @ V^T - Eq. 16
    dP = torch.bmm(dO, V.transpose(-2, -1))  # (batch, n_queries, n_keys)

    # dS = P * (dP - D) - Eq. 17
    dS = P * (dP - D.unsqueeze(-1))  # (batch, n_queries, n_keys)

    # dQ = dS @ K / sqrt(d) - Eq. 18
    dQ = torch.bmm(dS, K) * scale  # (batch, n_queries, d)

    # dK = dS^T @ Q / sqrt(d) - Eq. 19
    dK = torch.bmm(dS.transpose(-2, -1), Q) * scale  # (batch, n_keys, d)

    return dQ, dK, dV


class FlashAttention(torch.autograd.Function):
    """
    FlashAttention-2 autograd Function.

    Forward pass uses tiled online softmax. Backward pass uses recomputation with torch.compile.
    """

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        """
        Forward pass of FlashAttention-2.

        Args:
            ctx: Autograd context
            Q: Query tensor (batch, n_queries, d)
            K: Key tensor (batch, n_keys, d)
            V: Value tensor (batch, n_keys, d)
            is_causal: Whether to apply causal masking

        Returns:
            O: Output tensor (batch, n_queries, d)
        """
        O, L = flash_attention_forward_tiled(Q, K, V, is_causal=is_causal)
        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        """
        Backward pass of FlashAttention-2 using recomputation.
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_attention_backward_compiled(Q, K, V, O, dO, L, is_causal)
        return dQ, dK, dV, None
