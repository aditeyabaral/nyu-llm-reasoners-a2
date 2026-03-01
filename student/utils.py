"""
Utility functions for benchmarking.
"""

import torch
from contextlib import nullcontext

from student.config import BenchmarkConfig


def get_autocast_context(config: BenchmarkConfig):
    """Get the appropriate autocast context manager based on config."""
    if config.mixed_precision == "fp32":
        return nullcontext()
    elif config.mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    elif config.mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unknown mixed precision mode: {config.mixed_precision}")


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
