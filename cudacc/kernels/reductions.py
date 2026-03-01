"""
Reduction kernels: sum, min, max, histograms.

Implements parallel reduction algorithms for common statistical operations.
"""

from numba import cuda
import numpy as np
import math


@cuda.jit
def sum_kernel(array, output):
    """
    Parallel sum reduction kernel.
    
    Args:
        array: Input array on device.
        output: Output array (single element) on device.
    """
    # Shared memory for reduction
    shared = cuda.shared.array(shape=(256,), dtype=np.float32)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    
    idx = bid * block_size + tid
    
    # Load data into shared memory
    if idx < array.size:
        shared[tid] = array[idx]
    else:
        shared[tid] = 0.0
    
    cuda.syncthreads()
    
    # Perform reduction in shared memory
    s = block_size // 2
    while s > 0:
        if tid < s:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s //= 2
    
    # Write result for this block
    if tid == 0:
        cuda.atomic.add(output, 0, shared[0])


@cuda.jit
def min_kernel(array, output):
    """
    Parallel min reduction kernel.
    
    Args:
        array: Input array on device.
        output: Output array (single element) on device.
    """
    shared = cuda.shared.array(shape=(256,), dtype=np.float32)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    
    idx = bid * block_size + tid
    
    # Load data into shared memory
    if idx < array.size:
        shared[tid] = array[idx]
    else:
        shared[tid] = np.inf
    
    cuda.syncthreads()
    
    # Reduction
    s = block_size // 2
    while s > 0:
        if tid < s:
            shared[tid] = min(shared[tid], shared[tid + s])
        cuda.syncthreads()
        s //= 2
    
    if tid == 0:
        cuda.atomic.min(output, 0, shared[0])


@cuda.jit
def max_kernel(array, output):
    """
    Parallel max reduction kernel.
    
    Args:
        array: Input array on device.
        output: Output array (single element) on device.
    """
    shared = cuda.shared.array(shape=(256,), dtype=np.float32)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    
    idx = bid * block_size + tid
    
    # Load data into shared memory
    if idx < array.size:
        shared[tid] = array[idx]
    else:
        shared[tid] = -np.inf
    
    cuda.syncthreads()
    
    # Reduction
    s = block_size // 2
    while s > 0:
        if tid < s:
            shared[tid] = max(shared[tid], shared[tid + s])
        cuda.syncthreads()
        s //= 2
    
    if tid == 0:
        cuda.atomic.max(output, 0, shared[0])


@cuda.jit
def histogram_kernel(array, bins, hist):
    """
    Parallel histogram computation kernel.
    
    Args:
        array: Input array on device.
        bins: Bin edges on device.
        hist: Output histogram on device.
    """
    idx = cuda.grid(1)
    
    if idx < array.size:
        value = array[idx]
        
        # Binary search for bin
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                cuda.atomic.add(hist, i, 1)
                break


def gpu_sum(array):
    """
    Compute sum of array on GPU.
    
    Args:
        array: NumPy or CuPy array.
    
    Returns:
        Sum of all elements.
    """
    try:
        import cupy as cp
        device_array = cp.asarray(array)
        if device_array.size == 0:
            return 0.0
        output = cp.zeros(1, dtype=device_array.dtype)
        
        threads_per_block = 256
        blocks = math.ceil(device_array.size / threads_per_block)
        
        sum_kernel[blocks, threads_per_block](device_array.ravel(), output)
        
        return float(output[0])
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")


def gpu_min(array):
    """Compute minimum of array on GPU."""
    try:
        import cupy as cp
        device_array = cp.asarray(array)
        output = cp.full(1, np.inf, dtype=device_array.dtype)
        
        threads_per_block = 256
        blocks = math.ceil(device_array.size / threads_per_block)
        
        min_kernel[blocks, threads_per_block](device_array.ravel(), output)
        
        return float(output[0])
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")


def gpu_max(array):
    """Compute maximum of array on GPU."""
    try:
        import cupy as cp
        device_array = cp.asarray(array)
        output = cp.full(1, -np.inf, dtype=device_array.dtype)
        
        threads_per_block = 256
        blocks = math.ceil(device_array.size / threads_per_block)
        
        max_kernel[blocks, threads_per_block](device_array.ravel(), output)
        
        return float(output[0])
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")


def gpu_histogram(array, bins):
    """Compute histogram on GPU."""
    try:
        import cupy as cp
        device_array = cp.asarray(array)
        device_bins = cp.asarray(bins)
        hist = cp.zeros(len(bins) - 1, dtype=np.int32)
        
        threads_per_block = 256
        blocks = math.ceil(device_array.size / threads_per_block)
        
        histogram_kernel[blocks, threads_per_block](
            device_array.ravel(), device_bins, hist
        )
        
        return hist.get()
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")
