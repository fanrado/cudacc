"""
NumPy bridge: NumPy → CuPy/Numba replacements.

Intercepts NumPy operations and redirects them to GPU implementations.
"""

from typing import Any
from ..registry import PackageBridge
from ..kernels.reductions import gpu_sum, gpu_min, gpu_max
from ..kernels.transforms import gpu_multiply, gpu_add, gpu_normalize
import types


class NumpyBridge(PackageBridge):
    """
    Bridge for accelerating NumPy operations with CuPy and custom kernels.
    """
    
    def supports(self, pkg: Any) -> bool:
        """Check if this is the NumPy package."""
        return hasattr(pkg, '__name__') and pkg.__name__ == 'numpy'
    
    def apply(self, pkg: Any, devices: list[int]) -> Any:
        """
        Create an accelerated version of NumPy.
        
        Args:
            pkg: The numpy module.
            devices: List of GPU device IDs.
        
        Returns:
            Accelerated numpy-like module.
        """
        try:
            import cupy as cp
        except ImportError:
            raise RuntimeError("CuPy is required for NumPy acceleration")
        
        # Create a wrapper module
        accelerated = types.ModuleType('numpy_accelerated')
        accelerated.__doc__ = "GPU-accelerated NumPy via cudacc"
        
        # Set default device
        if devices:
            cp.cuda.Device(devices[0]).use()
        
        # Copy most functions from CuPy
        for attr_name in dir(cp):
            if not attr_name.startswith('_'):
                setattr(accelerated, attr_name, getattr(cp, attr_name))
        
        # Override specific functions with our custom kernels
        accelerated.sum = self._wrap_reduction(gpu_sum, cp.sum)
        accelerated.min = self._wrap_reduction(gpu_min, cp.min)
        accelerated.max = self._wrap_reduction(gpu_max, cp.max)
        accelerated.multiply = self._wrap_binary(gpu_multiply, cp.multiply)
        accelerated.add = self._wrap_binary(gpu_add, cp.add)
        
        # Add normalization function
        accelerated.normalize = gpu_normalize
        
        # Keep array creation on GPU
        accelerated.array = cp.array
        accelerated.zeros = cp.zeros
        accelerated.ones = cp.ones
        accelerated.empty = cp.empty
        
        # Store device info
        accelerated._cudacc_devices = devices
        accelerated._cudacc_backend = 'cupy+numba'
        
        return accelerated
    
    def _wrap_reduction(self, kernel_fn, fallback_fn):
        """
        Wrap a reduction kernel with fallback.
        
        Args:
            kernel_fn: Our custom kernel function.
            fallback_fn: CuPy fallback function.
        
        Returns:
            Wrapped function.
        """
        def wrapped(array, *args, **kwargs):
            try:
                # Try custom kernel for simple cases
                if len(args) == 0 and len(kwargs) == 0:
                    return kernel_fn(array)
            except Exception:
                pass
            
            # Fall back to CuPy
            return fallback_fn(array, *args, **kwargs)
        
        return wrapped
    
    def _wrap_binary(self, kernel_fn, fallback_fn):
        """Wrap a binary operation kernel with fallback."""
        def wrapped(a, b, *args, **kwargs):
            try:
                # Try custom kernel for simple cases
                if len(args) == 0 and len(kwargs) == 0:
                    return kernel_fn(a, b)
            except Exception:
                pass
            
            # Fall back to CuPy
            return fallback_fn(a, b, *args, **kwargs)
        
        return wrapped


# Register this bridge
from ..registry import register_bridge
register_bridge('numpy', NumpyBridge)
