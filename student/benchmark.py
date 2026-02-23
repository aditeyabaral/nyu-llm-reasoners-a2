#!/usr/bin/env python3
"""
Benchmarking script for the Basics Transformer Language Model.

This script benchmarks forward and backward passes for transformer models
with various configurations.
"""

import argparse
from html import parser
import timeit
from dataclasses import dataclass
import statistics

import torch
import torch.nn as nn

from a1_basics.model import BasicsTransformerLM


MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    # Model architecture
    vocab_size: int = 10000
    context_length: int = 512
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0

    # Benchmarking settings
    batch_size: int = 4
    warmup_steps: int = 5
    measurement_steps: int = 10
    forward_only: bool = False

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_model_size(cls, model_size: str, **kwargs):
        """Create config from a predefined model size."""
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")
        config = MODEL_CONFIGS[model_size].copy()
        config.update(kwargs)
        return cls(**config)


def print_gpu_specs():
    """Print GPU specifications if available."""
    if not torch.cuda.is_available():
        print("No CUDA devices available. Running on CPU.")
        return
    num_devices = torch.cuda.device_count()
    print(f"\n{num_devices} CUDA device(s) available")
    for i in range(num_devices):
        properties = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {properties.name}")
        print(f"    Total memory: {properties.total_memory / 1e9:.2f} GB")
        print(f"    Multiprocessors: {properties.multi_processor_count}")


def create_model(config: BenchmarkConfig) -> nn.Module:
    """Create a transformer model with the given configuration."""
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )
    model = model.to(config.device)
    return model


def create_random_batch(config: BenchmarkConfig) -> torch.Tensor:
    """Create a random batch of token IDs."""
    return torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=config.device,
    )


def benchmark_forward(model: nn.Module, input_ids: torch.Tensor, config: BenchmarkConfig) -> tuple[float, float]:
    """
    Benchmark the forward pass.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    times = []

    # Warmup
    for _ in range(config.warmup_steps):
        with torch.no_grad():
            _ = model(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Measurement
    for _ in range(config.measurement_steps):
        start = timeit.default_timer()
        with torch.no_grad():
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

    # Warmup
    for _ in range(config.warmup_steps):
        model.zero_grad()
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Measurement
    for _ in range(config.measurement_steps):
        model.zero_grad()
        start = timeit.default_timer()
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time


def run_benchmark(config: BenchmarkConfig):
    """Run the complete benchmark."""
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
    print(f"  forward_only:      {config.forward_only}")
    print(f"  device:            {config.device}")

    # Create model and data
    print("\nInitializing model...")
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    print("\nCreating random batch...")
    input_ids = create_random_batch(config)

    # Run benchmarks
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    # Forward pass
    print("\nBenchmarking forward pass...")
    fwd_mean, fwd_std = benchmark_forward(model, input_ids, config)
    print(f"  Forward pass: {fwd_mean:.2f} ± {fwd_std:.2f} ms")

    if not config.forward_only:
        # Forward + Backward pass
        print("\nBenchmarking forward + backward pass...")
        fwd_bwd_mean, fwd_bwd_std = benchmark_forward_backward(model, input_ids, config)
        print(f"  Forward + Backward: {fwd_bwd_mean:.2f} ± {fwd_bwd_std:.2f} ms")
        # Estimate backward-only time
        bwd_mean = fwd_bwd_mean - fwd_mean
        print(f"  Backward (estimated): {bwd_mean:.2f} ms")
    print("\n" + "=" * 70)

    return {
        "forward_mean": fwd_mean,
        "forward_std": fwd_std,
        "forward_backward_mean": fwd_bwd_mean if not config.forward_only else None,
        "forward_backward_std": fwd_bwd_std if not config.forward_only else None,
    }


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
    parser.add_argument("--forward-only", action="store_true", help="Only benchmark forward pass")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup steps (for comparison)")

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

    # Print GPU specs
    print_gpu_specs()

    # Build configuration
    if args.model_size:
        config_kwargs = MODEL_CONFIGS[args.model_size].copy()
    else:
        # Override with any explicitly provided arguments

        config_kwargs = {}
        config_kwargs["d_model"] = args.d_model
        config_kwargs["num_layers"] = args.num_layers
        config_kwargs["num_heads"] = args.num_heads
        config_kwargs["d_ff"] = args.d_ff

    config = BenchmarkConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        warmup_steps=0 if args.no_warmup else args.warmup_steps,
        measurement_steps=args.measurement_steps,
        forward_only=args.forward_only,
        device=args.device,
        **config_kwargs,
    )

    run_benchmark(config)
