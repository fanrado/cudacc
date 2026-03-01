"""
Core acceleration logic.

Provides the main accelerate() function that transparently patches
supported packages with GPU-accelerated implementations.
"""

from typing import Any, List, Optional
from .registry import get_package_bridge
from .utils.device import detect_devices, validate_devices


def accelerate(pkg: Any, devices: Optional[List[int]] = None) -> Any:
    """
    Accelerate a package by replacing CPU operations with GPU kernels.
    
    Args:
        pkg: The package or module to accelerate (e.g., numpy, scipy).
        devices: List of GPU device IDs to use. If None, uses all available GPUs.
    
    Returns:
        The accelerated package with GPU-backed operations.
    
    Raises:
        ValueError: If the package is not supported or devices are invalid.
        RuntimeError: If CUDA is not available.
    
    Example:
        >>> import numpy as np
        >>> from cudacc import accelerate
        >>> np_gpu = accelerate(np, devices=[0])
        >>> # Now np_gpu operations run on GPU 0
    """
    # Validate CUDA availability
    available_devices = detect_devices()
    if not available_devices:
        raise RuntimeError("No CUDA devices available")
    
    # Validate requested devices
    if devices is None:
        devices = available_devices
    else:
        validate_devices(devices, available_devices)
    
    # Get the bridge for this package
    bridge = get_package_bridge(pkg)
    if bridge is None:
        raise ValueError(f"Package {pkg.__name__} is not supported by cudacc")
    
    # Apply the bridge to create accelerated version
    accelerated_pkg = bridge.apply(pkg, devices)
    
    return accelerated_pkg
