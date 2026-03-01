"""
FlashAttention-2 benchmarking script.

Compares performance of Triton FlashAttention-2 vs regular PyTorch attention.
"""

import argparse
import itertools
import torch
import triton

from a1_basics.model import scaled_dot_product_attention
from student.utils import print_gpu_specs
from student.flash_attention_triton import FlashAttentionTriton


# Default configurations
SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
EMBED_DIMS = [16, 32, 64, 128]
DTYPES = ["bf16", "fp32"]
BATCH_SIZE = 1


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp32":
        return torch.float32
    elif dtype_str == "fp16":
        return torch.float16
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def create_qkv(batch_size: int, seq_len: int, d: int, device: str, dtype: torch.dtype):
    """Create random Q, K, V tensors."""
    Q = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    return Q, K, V


def benchmark_pytorch_attention(Q, K, V, is_causal=True):
    """Benchmark functions for PyTorch attention."""

    def forward_fn():
        return scaled_dot_product_attention(Q, K, V)

    def backward_fn():
        Q_b = Q.detach().clone().requires_grad_(True)
        K_b = K.detach().clone().requires_grad_(True)
        V_b = V.detach().clone().requires_grad_(True)
        out = scaled_dot_product_attention(Q_b, K_b, V_b)
        out.sum().backward()

    def forward_backward_fn():
        Q_b = Q.detach().clone().requires_grad_(True)
        K_b = K.detach().clone().requires_grad_(True)
        V_b = V.detach().clone().requires_grad_(True)
        out = scaled_dot_product_attention(Q_b, K_b, V_b)
        out.sum().backward()

    return forward_fn, backward_fn, forward_backward_fn


def benchmark_flash_attention(flash_fn, Q, K, V, is_causal=True):
    """Benchmark functions for FlashAttention."""

    def forward_fn():
        return flash_fn(Q, K, V, is_causal)

    def backward_fn():
        Q_b = Q.detach().clone().requires_grad_(True)
        K_b = K.detach().clone().requires_grad_(True)
        V_b = V.detach().clone().requires_grad_(True)
        out = flash_fn(Q_b, K_b, V_b, is_causal)
        out.sum().backward()

    def forward_backward_fn():
        Q_b = Q.detach().clone().requires_grad_(True)
        K_b = K.detach().clone().requires_grad_(True)
        V_b = V.detach().clone().requires_grad_(True)
        out = flash_fn(Q_b, K_b, V_b, is_causal)
        out.sum().backward()

    return forward_fn, backward_fn, forward_backward_fn


def run_benchmark_suite(
    seq_lengths: list[int] = SEQ_LENGTHS,
    embed_dims: list[int] = EMBED_DIMS,
    dtypes: list[str] = DTYPES,
    batch_size: int = BATCH_SIZE,
    is_causal: bool = True,
    warmup: int = 100,
    rep: int = 100,
    device: str = "cuda",
):
    """
    Run the full FlashAttention benchmark suite.

    Args:
        seq_lengths: Sequence lengths to benchmark
        embed_dims: Embedding dimensions to benchmark
        dtypes: Data types to benchmark ("bf16", "fp32")
        batch_size: Batch size (fixed to 1 per assignment)
        is_causal: Whether to use causal masking
        warmup: Warmup time in ms for triton.testing.do_bench
        rep: Repetition time in ms for triton.testing.do_bench
    """
    print("\n" + "=" * 100)
    print("FLASHATTENTION-2 BENCHMARKING")
    print("=" * 100)
    print("\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Embedding dimensions: {embed_dims}")
    print(f"  Data types: {dtypes}")
    print(f"  Causal: {is_causal}")
    print(f"  Warmup: {warmup} ms, Rep: {rep} ms")

    results = []

    # Print header
    print("\n" + "-" * 100)
    print(
        f"{'seq_len':>8} {'d':>6} {'dtype':>6} | {'PyT fwd':>10} {'PyT bwd':>10} {'PyT e2e':>10} | "
        f"{'FA fwd':>10} {'FA bwd':>10} {'FA e2e':>10} | {'status':>8}"
    )
    print("-" * 100)

    for seq_len, d, dtype_str in itertools.product(seq_lengths, embed_dims, dtypes):
        dtype = get_dtype(dtype_str)
        result = {
            "seq_len": seq_len,
            "d": d,
            "dtype": dtype_str,
            "batch_size": batch_size,
        }

        try:
            # Clear cache
            torch.cuda.empty_cache()

            # Create inputs
            Q, K, V = create_qkv(batch_size, seq_len, d, device, dtype)

            # Benchmark PyTorch attention
            pyt_fwd, pyt_bwd, pyt_e2e = benchmark_pytorch_attention(Q, K, V, is_causal)
            pyt_fwd_ms = triton.testing.do_bench(pyt_fwd, warmup=warmup, rep=rep)
            pyt_bwd_ms = triton.testing.do_bench(pyt_bwd, warmup=warmup, rep=rep)
            pyt_e2e_ms = triton.testing.do_bench(pyt_e2e, warmup=warmup, rep=rep)

            result["pyt_fwd_ms"] = pyt_fwd_ms
            result["pyt_bwd_ms"] = pyt_bwd_ms
            result["pyt_e2e_ms"] = pyt_e2e_ms

            # Benchmark FlashAttention
            fa_fwd, fa_bwd, fa_e2e = benchmark_flash_attention(FlashAttentionTriton.apply, Q, K, V, is_causal)
            fa_fwd_ms = triton.testing.do_bench(fa_fwd, warmup=warmup, rep=rep)
            fa_bwd_ms = triton.testing.do_bench(fa_bwd, warmup=warmup, rep=rep)
            fa_e2e_ms = triton.testing.do_bench(fa_e2e, warmup=warmup, rep=rep)

            result["fa_fwd_ms"] = fa_fwd_ms
            result["fa_bwd_ms"] = fa_bwd_ms
            result["fa_e2e_ms"] = fa_e2e_ms
            result["status"] = "OK"

            print(
                f"{seq_len:>8} {d:>6} {dtype_str:>6} | "
                f"{pyt_fwd_ms:>10.3f} {pyt_bwd_ms:>10.3f} {pyt_e2e_ms:>10.3f} | "
                f"{fa_fwd_ms:>10.3f} {fa_bwd_ms:>10.3f} {fa_e2e_ms:>10.3f} | "
                f"{'OK':>8}"
            )

            # Clean up
            del Q, K, V
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result["pyt_fwd_ms"] = float("nan")
                result["pyt_bwd_ms"] = float("nan")
                result["pyt_e2e_ms"] = float("nan")
                result["fa_fwd_ms"] = float("nan")
                result["fa_bwd_ms"] = float("nan")
                result["fa_e2e_ms"] = float("nan")
                result["status"] = "OOM"
                print(
                    f"{seq_len:>8} {d:>6} {dtype_str:>6} | "
                    f"{'--':>10} {'--':>10} {'--':>10} | "
                    f"{'--':>10} {'--':>10} {'--':>10} | "
                    f"{'OOM':>8}"
                )
                torch.cuda.empty_cache()
            else:
                raise

        results.append(result)

    print("-" * 100)

    # Print speedup summary
    print("\n" + "=" * 100)
    print("SPEEDUP SUMMARY (FlashAttention vs PyTorch)")
    print("=" * 100)
    print(f"\n{'seq_len':>8} {'d':>6} {'dtype':>6} | {'fwd speedup':>12} {'bwd speedup':>12} {'e2e speedup':>12}")
    print("-" * 60)

    for r in results:
        if r["status"] == "OK":
            fwd_speedup = r["pyt_fwd_ms"] / r["fa_fwd_ms"]
            bwd_speedup = r["pyt_bwd_ms"] / r["fa_bwd_ms"]
            e2e_speedup = r["pyt_e2e_ms"] / r["fa_e2e_ms"]
            print(
                f"{r['seq_len']:>8} {r['d']:>6} {r['dtype']:>6} | "
                f"{fwd_speedup:>11.2f}x {bwd_speedup:>11.2f}x {e2e_speedup:>11.2f}x"
            )
        else:
            print(f"{r['seq_len']:>8} {r['d']:>6} {r['dtype']:>6} | {'--':>12} {'--':>12} {'--':>12}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    oom_configs = [r for r in results if r["status"] == "OOM"]
    if oom_configs:
        print(f"\nOut-of-memory errors at {len(oom_configs)} configurations:")
        for r in oom_configs:
            print(f"  seq_len={r['seq_len']}, d={r['d']}, dtype={r['dtype']}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FlashAttention-2 vs PyTorch attention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=SEQ_LENGTHS,
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--embed-dims",
        type=int,
        nargs="+",
        default=EMBED_DIMS,
        help="Embedding dimensions to benchmark",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=DTYPES,
        help="Data types to benchmark (bf16, fp32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (fixed)",
    )
    parser.add_argument(
        "--no-causal",
        action="store_true",
        help="Disable causal masking",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup time in ms",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Repetition time in ms",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    print_gpu_specs()

    results = run_benchmark_suite(
        seq_lengths=args.seq_lengths,
        embed_dims=args.embed_dims,
        dtypes=args.dtypes,
        batch_size=args.batch_size,
        is_causal=not args.no_causal,
        warmup=args.warmup,
        rep=args.rep,
        device=args.device,
    )
