# Architecture

This document explains the internal design and architecture of cudacc.

## Overview

cudacc uses a **bridge pattern** to provide transparent GPU acceleration for different Python packages. The architecture consists of several key components that work together to intercept CPU operations and dispatch them to GPU kernels.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Application                        │
│                   (NumPy, SciPy, etc.)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    accelerate() Entry Point                  │
│                      (cudacc/__init__.py)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       Registry                               │
│              Maps packages to bridges                        │
│                  (cudacc/registry.py)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Package Bridges                           │
│      ┌──────────────┬──────────────┬──────────────┐        │
│      │ NumPy Bridge │ SciPy Bridge │ Uproot Bridge│        │
│      └──────────────┴──────────────┴──────────────┘        │
│                  (cudacc/bridges/)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Dispatcher                              │
│        Pattern matching → Kernel selection                   │
│                  (cudacc/dispatcher.py)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPU Kernels                               │
│      ┌──────────────┬──────────────┬──────────────┐        │
│      │  Reductions  │  Transforms  │   Physics    │        │
│      │ (sum, min,   │ (normalize,  │ (invariant   │        │
│      │  max, hist)  │  clip, etc.) │  mass, etc.) │        │
│      └──────────────┴──────────────┴──────────────┘        │
│                   (cudacc/kernels/)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Memory Management                          │
│              RMM-backed memory pooling                       │
│                  (cudacc/memory.py)                         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Entry Point (`accelerate()`)

The `accelerate()` function is the main API:

```python
def accelerate(pkg: Any, devices: Optional[List[int]] = None) -> Any:
    """
    Accelerate a package by replacing CPU operations with GPU kernels.
    """
```

**Responsibilities:**
- Validate CUDA availability
- Detect/validate GPU devices
- Look up the appropriate bridge for the package
- Apply the bridge to create an accelerated version

**Flow:**
1. Check CUDA availability
2. Validate requested devices
3. Query registry for package bridge
4. Apply bridge with specified devices
5. Return accelerated package

### 2. Registry

The registry maintains the mapping between packages and their bridges.

**Key Classes:**
- `PackageBridge`: Abstract base class for all bridges
- `PackageRegistry`: Maps package names to bridge classes

**Registration:**
```python
# Bridges self-register at import time
register_bridge('numpy', NumpyBridge)
register_bridge('scipy', ScipyBridge)
```

**Lookup:**
```python
bridge = get_package_bridge(numpy_module)
```

### 3. Bridges

Each supported package has a dedicated bridge that knows how to accelerate it.

#### Bridge Interface

```python
class PackageBridge(ABC):
    @abstractmethod
    def supports(self, pkg: Any) -> bool:
        """Check if this bridge supports the package."""
    
    @abstractmethod
    def apply(self, pkg: Any, devices: list[int]) -> Any:
        """Create accelerated version of the package."""
```

#### NumPy Bridge

**Strategy:** Create a wrapper module that:
1. Uses CuPy for most array operations
2. Overrides specific functions with custom Numba kernels
3. Maintains NumPy-like API

**Implementation:**
```python
class NumpyBridge(PackageBridge):
    def apply(self, pkg, devices):
        # Create wrapper module
        accelerated = types.ModuleType('numpy_accelerated')
        
        # Copy CuPy functions
        for attr in dir(cupy):
            setattr(accelerated, attr, getattr(cupy, attr))
        
        # Override with custom kernels
        accelerated.sum = wrap_kernel(gpu_sum)
        
        return accelerated
```

#### SciPy Bridge

**Strategy:** Directly use CuPy's `cupyx.scipy` module

**Implementation:**
- Maps SciPy submodules to CuPy equivalents
- Provides GPU-accelerated FFT, linalg, ndimage, etc.

#### Uproot Bridge

**Strategy:** Add HEP-specific GPU kernels to uproot/awkward

**Features:**
- Invariant mass calculations
- Particle filtering
- Lorentz transformations

### 4. Dispatcher

The dispatcher routes operations to appropriate kernels based on:
- Operation type (reduction, transform, etc.)
- Array properties (size, dtype, shape)
- Device capabilities

**Key Classes:**

```python
class KernelDispatcher:
    def register(self, operation, pattern, kernel, operation_type):
        """Register a dispatch rule."""
    
    def dispatch(self, operation, *args, **kwargs):
        """Find appropriate kernel for operation."""
```

**Pattern Matching:**

```python
def pattern_large_reduction(args):
    """Match reductions on large arrays."""
    arr = args[0][0]
    return arr.size > 100_000

dispatcher.register(
    "sum",
    pattern_large_reduction,
    gpu_sum_large,
    OperationType.REDUCTION
)
```

### 5. Kernels

GPU kernels are implemented using Numba CUDA.

#### Kernel Categories

**Reductions:**
- Sum, min, max
- Histograms
- Use parallel reduction patterns

**Transforms:**
- Element-wise operations
- Normalization
- Clipping
- Straightforward parallel execution

**Physics:**
- Invariant mass
- Particle distances
- Lorentz boosts
- Domain-specific algorithms

#### Example Kernel

```python
@cuda.jit
def sum_kernel(array, output):
    """Parallel sum reduction."""
    shared = cuda.shared.array(shape=(256,), dtype=np.float32)
    
    tid = cuda.threadIdx.x
    idx = cuda.grid(1)
    
    # Load to shared memory
    if idx < array.size:
        shared[tid] = array[idx]
    else:
        shared[tid] = 0.0
    
    cuda.syncthreads()
    
    # Reduction in shared memory
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s //= 2
    
    # Write result
    if tid == 0:
        cuda.atomic.add(output, 0, shared[0])
```

### 6. Memory Management

Efficient GPU memory management is crucial for performance.

#### Memory Pool

```python
class MemoryPool:
    """RMM-backed memory pool for a single device."""
    
    def allocate(self, size, dtype):
        """Allocate GPU memory."""
    
    def free_unused(self):
        """Free unused blocks."""
```

#### Multi-GPU Manager

```python
class MultiGPUMemoryManager:
    """Coordinate memory across multiple GPUs."""
    
    def __init__(self, device_ids):
        self.pools = {
            device_id: MemoryPool(device_id)
            for device_id in device_ids
        }
```

**Features:**
- Memory pooling to reduce allocation overhead
- RMM integration when available
- Fallback to CuPy memory pools
- Per-device pool management

### 7. Utilities

#### Device Management

```python
# Device detection
devices = detect_devices()

# Device properties
props = get_device_properties(0)

# Peer access matrix
matrix = get_peer_access_matrix()

# Select best device
best = select_best_device()
```

#### Profiling

```python
# Context manager
with GPUProfiler("operation") as profiler:
    result = compute_on_gpu(data)
    print(profiler.results)

# Decorator
@profile("my_function")
def my_function(data):
    return process(data)

# Benchmarking
results = benchmark(my_function, data, iterations=100)
```

## Data Flow

### Example: NumPy Sum Operation

```
1. User calls: np_gpu.sum(array)
   │
   ├─→ 2. Bridge wrapper checks if custom kernel available
   │       │
   │       ├─→ Yes: Use custom gpu_sum kernel
   │       │   │
   │       │   ├─→ 3. Kernel launcher calculates grid/block dims
   │       │   │
   │       │   ├─→ 4. Launch sum_kernel on GPU
   │       │   │
   │       │   └─→ 5. Return result
   │       │
   │       └─→ No: Fall back to CuPy implementation
   │           │
   │           └─→ 5. Return result
   │
   └─→ 6. User receives result
```

## Design Patterns

### 1. Bridge Pattern

**Purpose:** Decouple package-specific logic from core acceleration

**Benefits:**
- Easy to add new packages
- Package-specific optimizations
- Clean separation of concerns

### 2. Strategy Pattern (Dispatcher)

**Purpose:** Select kernel implementation at runtime

**Benefits:**
- Flexible kernel selection
- Multiple implementations per operation
- Easy to add new kernels

### 3. Registry Pattern

**Purpose:** Centralized package-to-bridge mapping

**Benefits:**
- Single source of truth
- Easy package discovery
- Extensible architecture

### 4. Factory Pattern (Memory Pools)

**Purpose:** Create appropriate memory pool for each device

**Benefits:**
- Consistent interface
- Device-specific configuration
- Resource management

## Extension Points

### Adding a New Package

1. **Create Bridge:**
   ```python
   class MyPackageBridge(PackageBridge):
       def supports(self, pkg):
           return pkg.__name__ == 'mypackage'
       
       def apply(self, pkg, devices):
           # Create accelerated version
           return accelerated_pkg
   ```

2. **Register Bridge:**
   ```python
   register_bridge('mypackage', MyPackageBridge)
   ```

3. **Add Kernels (optional):**
   ```python
   @cuda.jit
   def my_kernel(...):
       # Implementation
   ```

### Adding a New Kernel

1. **Implement Kernel:**
   ```python
   @cuda.jit
   def new_operation_kernel(input, output, params):
       idx = cuda.grid(1)
       if idx < input.size:
           output[idx] = compute(input[idx], params)
   ```

2. **Create Wrapper:**
   ```python
   def gpu_new_operation(data, params):
       data_gpu = cp.asarray(data)
       output = cp.empty_like(data_gpu)
       
       threads = 256
       blocks = math.ceil(data.size / threads)
       
       new_operation_kernel[blocks, threads](data_gpu, output, params)
       
       return output
   ```

3. **Register with Dispatcher (optional):**
   ```python
   dispatcher.register(
       "new_operation",
       pattern_fn,
       gpu_new_operation,
       OperationType.TRANSFORM
   )
   ```

## Performance Considerations

### 1. Kernel Launch Overhead

- Minimize kernel launches
- Batch operations when possible
- Use larger block sizes for large arrays

### 2. Memory Transfers

- Keep data on GPU
- Use pinned memory for faster transfers
- Stream data when possible

### 3. Occupancy

- Balance threads per block
- Consider shared memory usage
- Profile with nsys/nvprof

### 4. Algorithm Selection

- Different algorithms for different sizes
- Use dispatcher for automatic selection
- Benchmark on target hardware

## Testing Strategy

### Unit Tests

- Each kernel tested independently
- Bridge registration and lookup
- Dispatcher pattern matching

### Integration Tests

- End-to-end acceleration workflows
- Multi-GPU scenarios
- Memory management

### Performance Tests

- Benchmark against CPU
- Profile memory usage
- Test scaling across devices

## Future Enhancements

- **Just-In-Time Code Generation:** Generate kernels for specific array shapes
- **Automatic Batching:** Batch small operations together
- **Distributed Multi-GPU:** Automatic data distribution
- **More Packages:** pandas, scikit-learn, etc.
- **Custom Kernel DSL:** Higher-level kernel description language
