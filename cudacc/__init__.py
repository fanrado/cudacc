"""
cudacc - Transparent CUDA acceleration for scientific Python packages.

Public API entry point.
"""

from .accelerator import accelerate

__version__ = "0.1.0"
__all__ = ["accelerate"]
