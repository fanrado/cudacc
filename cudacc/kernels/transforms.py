"""
Transform kernels: element-wise operations, normalization.

Implements parallelizable transformations on arrays.
"""

from numba import cuda
import numpy as np
import math


@cuda.jit
def elementwise_multiply_kernel(a, b, out):
    """
    Element-wise multiplication kernel.
    
    Args:
        a: First input array on device.
        b: Second input array on device.
        out: Output array on device.
    """
    idx = cuda.grid(1)
    
    if idx < out.size:
        out[idx] = a[idx] * b[idx]


@cuda.jit
def elementwise_add_kernel(a, b, out):
    """
    Element-wise addition kernel.
    
    Args:
        a: First input array on device.
        b: Second input array on device.
        out: Output array on device.
    """
    idx = cuda.grid(1)
    
    if idx < out.size:
        out[idx] = a[idx] + b[idx]


@cuda.jit
def scalar_multiply_kernel(array, scalar, out):
    """
    Scalar multiplication kernel.
    
    Args:
        array: Input array on device.
        scalar: Scalar value.
        out: Output array on device.
    """
    idx = cuda.grid(1)
    
    if idx < out.size:
        out[idx] = array[idx] * scalar


@cuda.jit
def normalize_kernel(array, mean, std, out):
    """
    Normalization kernel (z-score).
    
    Args:
        array: Input array on device.
        mean: Mean value.
        std: Standard deviation.
        out: Output normalized array on device.
    """
    idx = cuda.grid(1)
    
    if idx < out.size:
        out[idx] = (array[idx] - mean) / std


@cuda.jit
def apply_function_kernel(array, out):
    """
    Apply a mathematical function element-wise.
    
    Args:
        array: Input array on device.
        out: Output array on device.
    """
    idx = cuda.grid(1)
    
    if idx < out.size:
        # Example: square root
        out[idx] = math.sqrt(abs(array[idx]))


@cuda.jit
def clip_kernel(array, min_val, max_val, out):
    """
    Clip array values to range [min_val, max_val].
    
    Args:
        array: Input array on device.
        min_val: Minimum value.
        max_val: Maximum value.
        out: Output array on device.
    """
    idx = cuda.grid(1)
    
    if idx < out.size:
        val = array[idx]
        if val < min_val:
            out[idx] = min_val
        elif val > max_val:
            out[idx] = max_val
        else:
            out[idx] = val


def gpu_multiply(a, b):
    """
    Element-wise multiplication on GPU.
    
    Args:
        a: First array.
        b: Second array.
    
    Returns:
        Product array.
    """
    try:
        import cupy as cp
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        out = cp.empty_like(a_gpu)
        
        threads_per_block = 256
        blocks = math.ceil(out.size / threads_per_block)
        
        elementwise_multiply_kernel[blocks, threads_per_block](
            a_gpu.ravel(), b_gpu.ravel(), out.ravel()
        )
        
        return out.reshape(a_gpu.shape)
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")


def gpu_add(a, b):
    """Element-wise addition on GPU."""
    try:
        import cupy as cp
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        out = cp.empty_like(a_gpu)
        
        threads_per_block = 256
        blocks = math.ceil(out.size / threads_per_block)
        
        elementwise_add_kernel[blocks, threads_per_block](
            a_gpu.ravel(), b_gpu.ravel(), out.ravel()
        )
        
        return out.reshape(a_gpu.shape)
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")


def gpu_normalize(array, mean=None, std=None):
    """
    Normalize array on GPU (z-score normalization).
    
    Args:
        array: Input array.
        mean: Mean value (computed if None).
        std: Standard deviation (computed if None).
    
    Returns:
        Normalized array.
    """
    try:
        import cupy as cp
        array_gpu = cp.asarray(array)
        
        if mean is None:
            mean = float(cp.mean(array_gpu))
        if std is None:
            std = float(cp.std(array_gpu))
        
        out = cp.empty_like(array_gpu)
        
        threads_per_block = 256
        blocks = math.ceil(out.size / threads_per_block)
        
        normalize_kernel[blocks, threads_per_block](
            array_gpu.ravel(), mean, std, out.ravel()
        )
        
        return out.reshape(array_gpu.shape)
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")


def gpu_clip(array, min_val, max_val):
    """Clip array values on GPU."""
    try:
        import cupy as cp
        array_gpu = cp.asarray(array)
        out = cp.empty_like(array_gpu)
        
        threads_per_block = 256
        blocks = math.ceil(out.size / threads_per_block)
        
        clip_kernel[blocks, threads_per_block](
            array_gpu.ravel(), min_val, max_val, out.ravel()
        )
        
        return out.reshape(array_gpu.shape)
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")
