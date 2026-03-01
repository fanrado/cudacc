"""
SciPy bridge: SciPy acceleration patches.

Provides GPU acceleration for SciPy operations using CuPy's scipy module.
"""

from typing import Any
from ..registry import PackageBridge
import types


class ScipyBridge(PackageBridge):
    """
    Bridge for accelerating SciPy operations with CuPy's scipy module.
    """
    
    def supports(self, pkg: Any) -> bool:
        """Check if this is the SciPy package."""
        return hasattr(pkg, '__name__') and pkg.__name__ == 'scipy'
    
    def apply(self, pkg: Any, devices: list[int]) -> Any:
        """
        Create an accelerated version of SciPy.
        
        Args:
            pkg: The scipy module.
            devices: List of GPU device IDs.
        
        Returns:
            Accelerated scipy-like module.
        """
        try:
            import cupy as cp
            import cupyx.scipy as cupyx_scipy
        except ImportError:
            raise RuntimeError("CuPy is required for SciPy acceleration")
        
        # Create wrapper module
        accelerated = types.ModuleType('scipy_accelerated')
        accelerated.__doc__ = "GPU-accelerated SciPy via cudacc"
        
        # Set default device
        if devices:
            cp.cuda.Device(devices[0]).use()
        
        # Map SciPy submodules to CuPy equivalents
        submodules = [
            'fft',
            'linalg',
            'ndimage',
            'signal',
            'sparse',
            'special',
            'stats',
        ]
        
        for submod in submodules:
            if hasattr(cupyx_scipy, submod):
                setattr(accelerated, submod, getattr(cupyx_scipy, submod))
        
        # Store device info
        accelerated._cudacc_devices = devices
        accelerated._cudacc_backend = 'cupy.scipy'
        
        return accelerated


# Register this bridge
from ..registry import register_bridge
register_bridge('scipy', ScipyBridge)
