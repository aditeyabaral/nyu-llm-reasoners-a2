"""
Benchmarking script for the Basics Transformer Language Model.

This script benchmarks forward and backward passes for transformer models with various configurations.
"""

import argparse
import timeit

import statistics

import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from a1_basics.optimizer import AdamW

from student.config import BenchmarkConfig, MODEL_CONFIGS
from student.utils import get_autocast_context, print_gpu_specs
from student.model import create_model, create_random_batch
from student.attention import enable_attention_profiling


def benchmark_forward(model: nn.Module, input_ids: torch.Tensor, config: BenchmarkConfig) -> tuple[float, float]:
    """
    Benchmark the forward pass.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    times = []
    autocast_ctx = get_autocast_context(config)

    # Warmup
    with nvtx.range("warmup_forward"):
        for _ in range(config.warmup_steps):
            with torch.no_grad(), autocast_ctx:
                _ = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Measurement
    for _ in range(config.measurement_steps):
        with nvtx.range("forward_iteration"):
            start = timeit.default_timer()
            with torch.no_grad(), autocast_ctx:
                _ = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = timeit.default_timer()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time


def benchmark_forward_backward(
    model: nn.Module, input_ids: torch.Tensor, config: BenchmarkConfig
) -> tuple[float, float]:
    """
    Benchmark forward and backward passes together.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    times = []
    autocast_ctx = get_autocast_context(config)

    # Warmup
    with nvtx.range("warmup_forward_backward"):
        for _ in range(config.warmup_steps):
            model.zero_grad()
            with autocast_ctx:
                logits = model(input_ids)
                loss = logits.sum()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Measurement
    for _ in range(config.measurement_steps):
        model.zero_grad()
        with nvtx.range("forward_backward_iteration"):
            start = timeit.default_timer()
            with nvtx.range("forward_pass"), autocast_ctx:
                logits = model(input_ids)
                loss = logits.sum()
            with nvtx.range("backward_pass"):
                loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = timeit.default_timer()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time


def benchmark_backward_only(model: nn.Module, input_ids: torch.Tensor, config: BenchmarkConfig) -> tuple[float, float]:
    """
    Benchmark the backward pass only, by running forward outside the timer
    to build the computation graph, then timing only loss.backward().

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    times = []
    autocast_ctx = get_autocast_context(config)

    # Warmup
    with nvtx.range("warmup_backward_only"):
        for _ in range(config.warmup_steps):
            model.zero_grad()
            with autocast_ctx:
                logits = model(input_ids)
                loss = logits.sum()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Measurement
    for _ in range(config.measurement_steps):
        model.zero_grad()
        # Forward pass outside the timer — just to build the computation graph
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.sum()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Time only the backward pass
        with nvtx.range("backward_only_iteration"):
            start = timeit.default_timer()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = timeit.default_timer()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time


def benchmark_training_step(model: nn.Module, input_ids: torch.Tensor, config: BenchmarkConfig) -> tuple[float, float]:
    """
    Benchmark a complete training step: forward + backward + optimizer step.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    times = []
    autocast_ctx = get_autocast_context(config)

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Warmup
    with nvtx.range("warmup_training_step"):
        for _ in range(config.warmup_steps):
            model.zero_grad()
            with autocast_ctx:
                logits = model(input_ids)
                loss = logits.sum()
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Measurement
    for _ in range(config.measurement_steps):
        with nvtx.range("training_step_iteration"):
            start = timeit.default_timer()
            model.zero_grad()
            with nvtx.range("forward_pass"), autocast_ctx:
                logits = model(input_ids)
                loss = logits.sum()
            with nvtx.range("backward_pass"):
                loss.backward()
            with nvtx.range("optimizer_step"):
                optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = timeit.default_timer()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time


def run_memory_profile(config: BenchmarkConfig, profile_mode: str, snapshot_path: str):
    """
    Run memory profiling and save snapshot for visualization.

    Args:
        config: Benchmark configuration
        profile_mode: "forward" or "training"
        snapshot_path: Path to save the memory snapshot pickle file
    """
    print("\n" + "=" * 70)
    print("MEMORY PROFILING")
    print("=" * 70)

    print("\nModel Configuration:")
    print(f"  vocab_size:     {config.vocab_size}")
    print(f"  context_length: {config.context_length}")
    print(f"  d_model:        {config.d_model}")
    print(f"  num_layers:     {config.num_layers}")
    print(f"  num_heads:      {config.num_heads}")
    print(f"  d_ff:           {config.d_ff}")

    print("\nMemory Profile Settings:")
    print(f"  profile_mode:    {profile_mode}")
    print(f"  mixed_precision: {config.mixed_precision}")
    print(f"  snapshot_path:   {snapshot_path}")

    # Create model and data
    print("\nInitializing model...")
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    print("\nCreating random batch...")
    input_ids = create_random_batch(config)

    autocast_ctx = get_autocast_context(config)

    # Warmup (before recording to get clean snapshot)
    print("\nRunning warmup...")
    if profile_mode == "forward":
        for _ in range(config.warmup_steps):
            with torch.no_grad(), autocast_ctx:
                _ = model(input_ids)
            torch.cuda.synchronize()
    else:  # training
        optimizer = AdamW(model.parameters(), lr=1e-4)
        for _ in range(config.warmup_steps):
            model.zero_grad()
            with autocast_ctx:
                logits = model(input_ids)
                loss = logits.sum()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

    # Clear memory stats before profiling
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Start recording memory history
    print("\nStarting memory recording...")
    torch.cuda.memory._record_memory_history(max_entries=1000000, stacks="all")

    # Run single iteration for profiling
    print(f"Running {profile_mode} pass...")
    if profile_mode == "forward":
        with torch.no_grad(), autocast_ctx:
            _ = model(input_ids)
        torch.cuda.synchronize()
    else:  # training
        if "optimizer" not in locals():
            optimizer = AdamW(model.parameters(), lr=1e-4)
        model.zero_grad()
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # Save memory snapshot
    print(f"\nSaving memory snapshot to {snapshot_path}...")
    torch.cuda.memory._dump_snapshot(snapshot_path)

    # Stop recording
    torch.cuda.memory._record_memory_history(enabled=None)

    # Report peak memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
    print(f"\nPeak memory usage: {peak_memory:.2f} MB")

    print("\n" + "=" * 70)
    print(f"Memory snapshot saved to: {snapshot_path}")
    print("=" * 70)

    return {"peak_memory_mb": peak_memory}


def run_benchmark(config: BenchmarkConfig, profile_mode: str = "all", use_compile: bool = False):
    """
    Run the complete benchmark.

    Args:
        config: Benchmark configuration
        profile_mode: One of "forward", "backward", "training", "all"
        use_compile: Whether to compile the model with torch.compile
    """
    print("\n" + "=" * 70)
    print("TRANSFORMER MODEL BENCHMARKING")
    print("=" * 70)

    # Print configuration
    print("\nModel Configuration:")
    print(f"  vocab_size:     {config.vocab_size}")
    print(f"  context_length: {config.context_length}")
    print(f"  d_model:        {config.d_model}")
    print(f"  num_layers:     {config.num_layers}")
    print(f"  num_heads:      {config.num_heads}")
    print(f"  d_ff:           {config.d_ff}")
    print(f"  rope_theta:     {config.rope_theta}")

    print("\nBenchmark Settings:")
    print(f"  batch_size:        {config.batch_size}")
    print(f"  warmup_steps:      {config.warmup_steps}")
    print(f"  measurement_steps: {config.measurement_steps}")
    print(f"  mixed_precision:   {config.mixed_precision}")
    print(f"  compile:           {use_compile}")
    print(f"  profile_mode:      {profile_mode}")
    print(f"  device:            {config.device}")

    # Create model and data
    print("\nInitializing model...")
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Compile model if requested
    if use_compile:
        print("\nCompiling model with torch.compile...")
        model = torch.compile(model)

    print("\nCreating random batch...")
    input_ids = create_random_batch(config)

    # Run benchmarks
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    results = {}

    # Forward pass
    if profile_mode in ("forward", "all"):
        print("\nBenchmarking forward pass...")
        fwd_mean, fwd_std = benchmark_forward(model, input_ids, config)
        print(f"  Forward pass: {fwd_mean:.2f} ± {fwd_std:.2f} ms")
        results["forward_mean"] = fwd_mean
        results["forward_std"] = fwd_std

    # Forward + Backward pass
    if profile_mode in ("backward", "all"):
        print("\nBenchmarking forward + backward pass...")
        fwd_bwd_mean, fwd_bwd_std = benchmark_forward_backward(model, input_ids, config)
        print(f"  Forward + Backward: {fwd_bwd_mean:.2f} ± {fwd_bwd_std:.2f} ms")
        results["forward_backward_mean"] = fwd_bwd_mean
        results["forward_backward_std"] = fwd_bwd_std

        if "forward_mean" in results:
            bwd_mean = fwd_bwd_mean - results["forward_mean"]
            print(f"  Backward (estimated): {bwd_mean:.2f} ms")

        print("\nBenchmarking backward pass only...")
        bwd_only_mean, bwd_only_std = benchmark_backward_only(model, input_ids, config)
        print(f"  Backward only: {bwd_only_mean:.2f} ± {bwd_only_std:.2f} ms")
        results["backward_only_mean"] = bwd_only_mean
        results["backward_only_std"] = bwd_only_std

    # Full training step
    if profile_mode in ("training", "all"):
        print("\nBenchmarking full training step (forward + backward + optimizer)...")
        train_mean, train_std = benchmark_training_step(model, input_ids, config)
        print(f"  Training step: {train_mean:.2f} ± {train_std:.2f} ms")
        results["training_step_mean"] = train_mean
        results["training_step_std"] = train_std

    print("\n" + "=" * 70)
    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer model forward and backward passes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model size presets
    parser.add_argument(
        "--model-size",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Use a predefined model size configuration.",
    )

    # Custom model parameters (override preset or use directly)
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=512, help="Context/sequence length")
    parser.add_argument("--d_model", type=int, default=None, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=None, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Benchmarking settings
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--measurement-steps", type=int, default=10, help="Number of measurement steps")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup steps (for comparison)")

    # Mixed precision
    parser.add_argument(
        "--mixed-precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Mixed precision mode: fp32 (full precision), fp16, or bf16",
    )

    # Compile
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile",
    )

    # Profile mode
    parser.add_argument(
        "--profile-mode",
        type=str,
        choices=["forward", "backward", "training", "all"],
        default="all",
        help="What to benchmark: forward only, forward+backward, full training step, or all",
    )

    # Attention profiling
    parser.add_argument(
        "--profile-attention",
        action="store_true",
        help="Enable detailed NVTX profiling of attention components (softmax, matmul)",
    )

    # Memory profiling
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Run memory profiling instead of timing benchmarks",
    )
    parser.add_argument(
        "--memory-snapshot-path",
        type=str,
        default="memory_snapshot.pickle",
        help="Path to save memory snapshot pickle file",
    )

    # Device and seed
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser, parser.parse_args()


if __name__ == "__main__":
    parser, args = parse_args()
    if args.model_size is None and None in (args.d_model, args.num_layers, args.num_heads, args.d_ff):
        parser.error("Must provide --model-size OR all of: --d_model, --num-layers, --num-heads, --d_ff")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    # Enable attention profiling if requested
    if args.profile_attention:
        print("Enabling detailed attention profiling...")
        enable_attention_profiling()

    # Print GPU specs
    print_gpu_specs()

    # Build configuration
    if args.model_size:
        config_kwargs = MODEL_CONFIGS[args.model_size].copy()
    else:
        config_kwargs = {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
        }

    config = BenchmarkConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        warmup_steps=0 if args.no_warmup else args.warmup_steps,
        measurement_steps=args.measurement_steps,
        mixed_precision=args.mixed_precision,
        device=args.device,
        **config_kwargs,
    )

    # Run memory profiling or regular benchmarking
    if args.profile_memory:
        if args.profile_mode not in ("forward", "training"):
            print("Warning: Memory profiling only supports 'forward' or 'training' modes. Defaulting to 'forward'.")
            profile_mode = "forward"
        else:
            profile_mode = args.profile_mode
        run_memory_profile(config, profile_mode, args.memory_snapshot_path)
    else:
        run_benchmark(config, profile_mode=args.profile_mode, use_compile=args.compile)
