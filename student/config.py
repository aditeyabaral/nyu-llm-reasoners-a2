"""
Configuration for benchmarking the transformer model.
"""

import torch
from dataclasses import dataclass


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

    # Mixed precision settings
    mixed_precision: str = "fp32"  # "fp32", "fp16", "bf16"

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
