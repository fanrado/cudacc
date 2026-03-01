# Quick Start Guide

Get started with cudacc in minutes.

## Basic Usage

**Note:** For array creation, create arrays on CPU using NumPy, then transfer to GPU using `np_gpu.array()`. This avoids potential CUDA library dependencies for random number generation while keeping all computations on GPU.

### 1. Accelerating NumPy

The simplest way to use cudacc:

```python
import numpy as np
from cudacc import accelerate

# Accelerate NumPy for GPU 0
np_gpu = accelerate(np, devices=[0])

# Create data on CPU, transfer to GPU
data_cpu = np.random.randn(10_000_000).astype(np.float32)
data = np_gpu.array(data_cpu)

# Perform operations on GPU
result = np_gpu.sum(data)
print(f"Sum: {result}")
```

### 2. Auto Device Selection

Let cudacc automatically select available GPUs:

```python
from cudacc import accelerate
import numpy as np

# Auto-detect and use all available GPUs
np_gpu = accelerate(np)

# Create array on CPU, transfer to GPU
arr_cpu = np.zeros(1000000, dtype=np.float32)
arr = np_gpu.array(arr_cpu)
```

### 3. Multiple Operations

Chain multiple operations:

```python
from cudacc import accelerate
import numpy as np

np_gpu = accelerate(np, devices=[0])

# Create data on CPU, transfer to GPU
data_cpu = np.random.randn(1_000_000).astype(np.float32)
data = np_gpu.array(data_cpu)

# Multiple operations (all on GPU)
normalized = (data - np_gpu.mean(data)) / np_gpu.std(data)
clipped = np_gpu.clip(normalized, -3, 3)
result = np_gpu.sum(clipped ** 2)

print(f"Result: {result}")
```

## Device Management

### Listing Devices

```python
from cudacc.utils.device import print_device_info

# Print information about all GPUs
print_device_info()
```

Output:
```
Found 2 CUDA device(s):

Device 0: NVIDIA GeForce RTX 3090
  Compute Capability: 8.6
  Total Memory: 24.00 GB
  Multiprocessors: 82
  Max Threads/Block: 1024

Device 1: NVIDIA GeForce RTX 3080
  Compute Capability: 8.6
  Total Memory: 10.00 GB
  Multiprocessors: 68
  Max Threads/Block: 1024
```

### Selecting Best Device

```python
from cudacc.utils.device import select_best_device

# Automatically select device with most free memory
best_device = select_best_device()
print(f"Using GPU {best_device}")
```

### Manual Device Control

```python
from cudacc.utils.device import set_device

# Manually set the active device
set_device(1)  # Use GPU 1
```

## Profiling

### Basic Profiling

```python
from cudacc.utils.profiler import GPUProfiler
from cudacc import accelerate
import numpy as np

np_gpu = accelerate(np, devices=[0])

# Profile an operation
with GPUProfiler("sum_operation") as profiler:
    data_cpu = np.random.randn(10_000_000).astype(np.float32)
    data = np_gpu.array(data_cpu)
    result = np_gpu.sum(data)

print(profiler.results)
```

### Function Profiling Decorator

```python
from cudacc.utils.profiler import profile
from cudacc import accelerate
import numpy as np

np_gpu = accelerate(np, devices=[0])

@profile("my_computation")
def my_function(n):
    data_cpu = np.random.randn(n).astype(np.float32)
    data = np_gpu.array(data_cpu)
    return np_gpu.sum(data ** 2)

result = my_function(10_000_000)
```

### Memory Tracking

```python
from cudacc.utils.profiler import MemoryTracker
from cudacc import accelerate
import numpy as np

np_gpu = accelerate(np, devices=[0])

tracker = MemoryTracker()

tracker.snapshot("Initial")

data1 = np_gpu.array(np.zeros(10_000_000, dtype=np.float32))
tracker.snapshot("After data1")

data2 = np_gpu.array(np.ones(10_000_000, dtype=np.float32))
tracker.snapshot("After data2")

tracker.print_summary()
```

## HEP-Specific Usage

### Invariant Mass Calculation

```python
from cudacc.kernels.physics import gpu_invariant_mass
import numpy as np

# Particle four-momenta (E, px, py, pz)
n = 100000

# First particles
e1 = np.random.uniform(1, 100, n).astype(np.float32)
px1 = np.random.randn(n).astype(np.float32) * 10
py1 = np.random.randn(n).astype(np.float32) * 10
pz1 = np.random.randn(n).astype(np.float32) * 20

# Second particles
e2 = np.random.uniform(1, 100, n).astype(np.float32)
px2 = np.random.randn(n).astype(np.float32) * 10
py2 = np.random.randn(n).astype(np.float32) * 10
pz2 = np.random.randn(n).astype(np.float32) * 20

# Compute invariant masses on GPU
masses = gpu_invariant_mass(e1, px1, py1, pz1, e2, px2, py2, pz2)

print(f"Mean invariant mass: {masses.mean()}")
```

## SciPy Acceleration

```python
from cudacc import accelerate
import scipy

# Accelerate SciPy
scipy_gpu = accelerate(scipy, devices=[0])

# Use scipy_gpu for accelerated operations
# (requires CuPy with scipy support)
import cupy as cp

data = cp.random.randn(1000, 1000).astype(cp.float32)

# FFT on GPU
if hasattr(scipy_gpu, 'fft'):
    fft_result = scipy_gpu.fft.fft2(data)
    print(f"FFT computed on GPU: {fft_result.shape}")
```

## Performance Comparison

Compare CPU vs GPU performance:

```python
import numpy as np
from cudacc import accelerate
import time

# Create test data
size = 50_000_000
cpu_data = np.random.randn(size).astype(np.float32)

# CPU version
start = time.time()
cpu_result = np.sum(cpu_data ** 2)
cpu_time = time.time() - start

# GPU version
np_gpu = accelerate(np, devices=[0])
gpu_data = np_gpu.array(cpu_data)

start = time.time()
gpu_result = np_gpu.sum(gpu_data ** 2)
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f}s")
print(f"GPU time: {gpu_time:.4f}s")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
print(f"Results match: {np.isclose(cpu_result, float(gpu_result))}")
```

## Memory Management

### Using Memory Pools

```python
from cudacc.memory import MemoryPool

# Create memory pool for device 0
pool = MemoryPool(device_id=0, pool_size=1024**3)  # 1 GB pool

# Allocate arrays
arr1 = pool.allocate(1_000_000)
arr2 = pool.allocate(1_000_000)

# Free unused memory
pool.free_unused()
```

### Multi-GPU Memory Management

```python
from cudacc.memory import MultiGPUMemoryManager

# Manage memory across multiple GPUs
manager = MultiGPUMemoryManager([0, 1])

# Get pool for specific device
pool_0 = manager.get_pool(0)
pool_1 = manager.get_pool(1)

# Allocate on different devices
arr_0 = pool_0.allocate(1_000_000)
arr_1 = pool_1.allocate(1_000_000)

# Clean up all devices
manager.free_all_unused()
```

## Best Practices

### 1. Minimize Data Transfers

Keep data on GPU as long as possible:

```python
# ✗ Bad: Transfers data back and forth
for i in range(100):
    cpu_data = np.array(gpu_data)  # Transfer to CPU
    gpu_data = np_gpu.array(cpu_data * 2)  # Transfer to GPU

# ✓ Good: Keep data on GPU
gpu_data = np_gpu.array(data)
for i in range(100):
    gpu_data = gpu_data * 2
```

### 2. Use Appropriate Array Sizes

GPU acceleration is most effective for large arrays:

```python
# Small arrays: CPU might be faster
small = np_gpu.array([1, 2, 3])  # Overhead may exceed benefit

# Large arrays: GPU shines
large_cpu = np.random.randn(10_000_000).astype(np.float32)
large = np_gpu.array(large_cpu)  # Great for GPU
```

### 3. Batch Operations

Combine operations when possible:

```python
# ✓ Good: Combined operation
result = np_gpu.sum((data - mean) / std)

# ✗ Less efficient: Separate operations with sync
centered = data - mean
normalized = centered / std
result = np_gpu.sum(normalized)
```

## Next Steps

- Explore the [Examples](../examples/) directory
- Read the [Architecture](architecture.md) documentation
- Check the [API Reference](api/) for detailed function documentation
- Run the benchmark examples to see performance on your hardware
