import torch
import torch.nn as nn
from student.config import BenchmarkConfig

from a1_basics.model import BasicsTransformerLM


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
