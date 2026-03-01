"""
Basic acceleration example.

Demonstrates a simple before/after comparison of NumPy operations
with and without GPU acceleration.
"""

import numpy as np
import time


def cpu_computation(data):
    """Perform computation on CPU."""
    # Normalize data
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std
    
    # Compute some statistics
    result = {
        'sum': np.sum(normalized),
        'min': np.min(normalized),
        'max': np.max(normalized),
        'mean': np.mean(normalized),
    }
    
    return result


def gpu_computation(data):
    """Perform computation on GPU using cudacc."""
    from cudacc import accelerate
    
    # Accelerate NumPy
    np_gpu = accelerate(np, devices=[0])
    
    # Transfer data to GPU
    data_gpu = np_gpu.array(data)
    
    # Normalize data (on GPU)
    normalized = np_gpu.normalize(data_gpu)
    
    # Compute statistics (on GPU)
    result = {
        'sum': float(np_gpu.sum(normalized)),
        'min': float(np_gpu.min(normalized)),
        'max': float(np_gpu.max(normalized)),
        'mean': float(np_gpu.mean(normalized)),
    }
    
    return result


def main():
    """Run the comparison."""
    print("=" * 60)
    print("cudacc Basic Acceleration Example")
    print("=" * 60)
    print()
    
    # Generate large dataset
    size = 10_000_000
    print(f"Generating {size:,} random numbers...")
    data = np.random.randn(size).astype(np.float32)
    print(f"Data size: {data.nbytes / 1024**2:.2f} MB")
    print()
    
    # CPU computation
    print("Running on CPU...")
    start = time.time()
    cpu_result = cpu_computation(data)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"CPU result: {cpu_result}")
    print()
    
    # GPU computation
    try:
        print("Running on GPU with cudacc...")
        start = time.time()
        gpu_result = gpu_computation(data)
        gpu_time = time.time() - start
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"GPU result: {gpu_result}")
        print()
        
        # Speedup
        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.2f}x")
        print()
        
        # Verify results match
        print("Verifying results...")
        for key in cpu_result:
            cpu_val = cpu_result[key]
            gpu_val = gpu_result[key]
            match = np.isclose(cpu_val, gpu_val, rtol=1e-4)
            print(f"  {key}: CPU={cpu_val:.6f}, GPU={gpu_val:.6f}, Match={match}")
        
    except Exception as e:
        print(f"GPU computation failed: {e}")
        print("Make sure you have CUDA and CuPy installed.")
    
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
