"""
Attention benchmarking script.

Benchmarks PyTorch attention at different scales (head dimensions and sequence lengths).
Supports both compiled and uncompiled versions.
Sections 1.2.1 and 1.3(a) of the assignment.
"""

import argparse
import itertools
import torch

from a1_basics.model import scaled_dot_product_attention
from student.utils import print_gpu_specs


# Default configurations
HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENGTHS = [256, 1024, 4096, 8192, 16384]
BATCH_SIZE = 8
NUM_ITERATIONS = 100
WARMUP_ITERATIONS = 10


def create_qkv(batch_size: int, seq_len: int, d_model: int, device: str) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, V tensors."""
    Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    return Q, K, V


def benchmark_attention_forward(
    attention_fn: callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_iterations: int,
    warmup_iterations: int,
) -> float:
    """
    Benchmark forward pass of attention.

    Returns:
        Mean time per iteration in milliseconds.
    """
    # Warmup
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = attention_fn(Q, K, V)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = attention_fn(Q, K, V)
        torch.cuda.synchronize()
    end.record()

    torch.cuda.synchronize()
    total_time_ms = start.elapsed_time(end)
    return total_time_ms / num_iterations


def benchmark_attention_backward(
    attention_fn: callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_iterations: int,
    warmup_iterations: int,
) -> tuple[float, float]:
    """
    Benchmark backward pass of attention.

    Returns:
        Tuple of (mean_time_ms, memory_before_backward_mb)
    """
    # Warmup
    for _ in range(warmup_iterations):
        Q_w = Q.detach().clone().requires_grad_(True)
        K_w = K.detach().clone().requires_grad_(True)
        V_w = V.detach().clone().requires_grad_(True)
        output = attention_fn(Q_w, K_w, V_w)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

    # Clear cache and measure memory before backward
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run forward to set up for backward
    Q_m = Q.detach().clone().requires_grad_(True)
    K_m = K.detach().clone().requires_grad_(True)
    V_m = V.detach().clone().requires_grad_(True)
    output = attention_fn(Q_m, K_m, V_m)
    torch.cuda.synchronize()

    # Memory before backward
    memory_before_backward = torch.cuda.memory_allocated() / (1024**2)  # MB

    # Benchmark backward
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(num_iterations):
        # Need fresh forward pass each time for clean backward
        if i > 0:
            Q_m = Q.detach().clone().requires_grad_(True)
            K_m = K.detach().clone().requires_grad_(True)
            V_m = V.detach().clone().requires_grad_(True)
            output = attention_fn(Q_m, K_m, V_m)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()
    end.record()

    torch.cuda.synchronize()
    total_time_ms = start.elapsed_time(end)

    return total_time_ms / num_iterations, memory_before_backward


def run_attention_benchmark(
    attention_fn: callable,
    head_dims: list[int] = HEAD_DIMS,
    seq_lengths: list[int] = SEQ_LENGTHS,
    batch_size: int = BATCH_SIZE,
    num_iterations: int = NUM_ITERATIONS,
    warmup_iterations: int = WARMUP_ITERATIONS,
    device: str = "cuda",
    label: str = "attention",
):
    """
    Run attention benchmarks across all configurations.

    Returns:
        List of result dictionaries.
    """
    print("\n" + "-" * 90)
    print(f"{'d_model':>8} {'seq_len':>10} {'fwd (ms)':>12} {'bwd (ms)':>12} {'mem (MB)':>12} {'status':>10}")
    print("-" * 90)

    results = []

    for d_model, seq_len in itertools.product(head_dims, seq_lengths):
        result = {
            "d_model": d_model,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "label": label,
        }

        try:
            # Clear cache before each config
            torch.cuda.empty_cache()

            # Create inputs
            Q, K, V = create_qkv(batch_size, seq_len, d_model, device)

            # Benchmark forward
            fwd_time = benchmark_attention_forward(attention_fn, Q, K, V, num_iterations, warmup_iterations)
            result["forward_ms"] = fwd_time

            # Benchmark backward
            bwd_time, mem_before_bwd = benchmark_attention_backward(
                attention_fn, Q, K, V, num_iterations, warmup_iterations
            )
            result["backward_ms"] = bwd_time
            result["memory_mb"] = mem_before_bwd
            result["status"] = "OK"

            print(f"{d_model:>8} {seq_len:>10} {fwd_time:>12.3f} {bwd_time:>12.3f} {mem_before_bwd:>12.2f} {'OK':>10}")

            # Clean up
            del Q, K, V
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result["forward_ms"] = float("nan")
                result["backward_ms"] = float("nan")
                result["memory_mb"] = float("nan")
                result["status"] = "OOM"
                print(f"{d_model:>8} {seq_len:>10} {'--':>12} {'--':>12} {'--':>12} {'OOM':>10}")
                torch.cuda.empty_cache()
            else:
                raise

        results.append(result)

    print("-" * 90)
    return results


def run_benchmark_suite(
    head_dims: list[int] = HEAD_DIMS,
    seq_lengths: list[int] = SEQ_LENGTHS,
    batch_size: int = BATCH_SIZE,
    num_iterations: int = NUM_ITERATIONS,
    warmup_iterations: int = WARMUP_ITERATIONS,
    device: str = "cuda",
    use_compile: bool = False,
):
    """
    Run the full benchmark suite.

    Args:
        use_compile: If True, also benchmark compiled attention.
    """
    print("\n" + "=" * 90)
    print("ATTENTION BENCHMARKING")
    print("=" * 90)
    print("\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Head dimensions: {head_dims}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Compile: {use_compile}")

    all_results = []

    # Benchmark uncompiled attention
    print("\n" + "=" * 90)
    print("UNCOMPILED ATTENTION")
    print("=" * 90)
    uncompiled_results = run_attention_benchmark(
        attention_fn=scaled_dot_product_attention,
        head_dims=head_dims,
        seq_lengths=seq_lengths,
        batch_size=batch_size,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
        device=device,
        label="uncompiled",
    )
    all_results.extend(uncompiled_results)

    # Benchmark compiled attention if requested
    if use_compile:
        print("\n" + "=" * 90)
        print("COMPILED ATTENTION (torch.compile)")
        print("=" * 90)

        compiled_attention = torch.compile(scaled_dot_product_attention)

        compiled_results = run_attention_benchmark(
            attention_fn=compiled_attention,
            head_dims=head_dims,
            seq_lengths=seq_lengths,
            batch_size=batch_size,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            device=device,
            label="compiled",
        )
        all_results.extend(compiled_results)

        # Print comparison summary
        print("\n" + "=" * 90)
        print("COMPARISON: COMPILED vs UNCOMPILED")
        print("=" * 90)
        print(f"\n{'d_model':>8} {'seq_len':>10} {'fwd speedup':>14} {'bwd speedup':>14}")
        print("-" * 50)

        for uc, c in zip(uncompiled_results, compiled_results):
            if uc["status"] == "OK" and c["status"] == "OK":
                fwd_speedup = uc["forward_ms"] / c["forward_ms"]
                bwd_speedup = uc["backward_ms"] / c["backward_ms"]
                print(f"{uc['d_model']:>8} {uc['seq_len']:>10} {fwd_speedup:>13.2f}x {bwd_speedup:>13.2f}x")
            else:
                print(f"{uc['d_model']:>8} {uc['seq_len']:>10} {'--':>14} {'--':>14}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    oom_configs = [r for r in all_results if r["status"] == "OOM"]
    if oom_configs:
        print(f"\nOut-of-memory errors at {len(oom_configs)} configurations:")
        for r in oom_configs:
            print(f"  [{r['label']}] d_model={r['d_model']}, seq_len={r['seq_len']}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch attention at different scales.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--head-dims",
        type=int,
        nargs="+",
        default=HEAD_DIMS,
        help="Head embedding dimensions to benchmark",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=SEQ_LENGTHS,
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (fixed)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=NUM_ITERATIONS,
        help="Number of iterations for timing",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=WARMUP_ITERATIONS,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also benchmark compiled attention (torch.compile)",
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
        head_dims=args.head_dims,
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
        device=args.device,
        use_compile=args.compile,
    )
