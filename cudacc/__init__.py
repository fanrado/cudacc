"""
cudacc - Transparent CUDA acceleration for scientific Python packages.

Public API entry point.
"""

from .accelerator import accelerate

# Import bridges to trigger self-registration
from . import bridges

__version__ = "0.1.0"
__all__ = ["accelerate"]
