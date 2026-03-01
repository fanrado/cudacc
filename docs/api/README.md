# API Reference

This directory contains the auto-generated API reference documentation for cudacc.

## Core API

### Main Entry Point

- **[accelerate()](accelerate.md)** - Main function to accelerate a package

### Registry

- **[PackageBridge](registry.md#packagebridge)** - Base class for package bridges
- **[PackageRegistry](registry.md#packageregistry)** - Registry of package bridges
- **[register_bridge()](registry.md#register_bridge)** - Register a new bridge

### Dispatcher

- **[KernelDispatcher](dispatcher.md#kerneldispatcher)** - Kernel dispatch manager
- **[DispatchRule](dispatcher.md#dispatchrule)** - Dispatch rule
- **[OperationType](dispatcher.md#operationtype)** - Operation categories

## Kernels

### Reductions

- **[gpu_sum()](kernels/reductions.md#gpu_sum)** - GPU sum reduction
- **[gpu_min()](kernels/reductions.md#gpu_min)** - GPU min reduction
- **[gpu_max()](kernels/reductions.md#gpu_max)** - GPU max reduction
- **[gpu_histogram()](kernels/reductions.md#gpu_histogram)** - GPU histogram

### Transforms

- **[gpu_multiply()](kernels/transforms.md#gpu_multiply)** - Element-wise multiplication
- **[gpu_add()](kernels/transforms.md#gpu_add)** - Element-wise addition
- **[gpu_normalize()](kernels/transforms.md#gpu_normalize)** - Array normalization
- **[gpu_clip()](kernels/transforms.md#gpu_clip)** - Value clipping

### Physics

- **[gpu_invariant_mass()](kernels/physics.md#gpu_invariant_mass)** - Invariant mass calculation
- **[gpu_particle_distance()](kernels/physics.md#gpu_particle_distance)** - Particle distance calculation

## Memory Management

- **[MemoryPool](memory.md#memorypool)** - Single-device memory pool
- **[MultiGPUMemoryManager](memory.md#multigpumemorymanager)** - Multi-GPU memory manager

## Utilities

### Device Management

- **[detect_devices()](utils/device.md#detect_devices)** - Detect available CUDA devices
- **[get_device_properties()](utils/device.md#get_device_properties)** - Get device properties
- **[select_best_device()](utils/device.md#select_best_device)** - Select best available device
- **[set_device()](utils/device.md#set_device)** - Set active device
- **[print_device_info()](utils/device.md#print_device_info)** - Print device information

### Profiling

- **[GPUProfiler](utils/profiler.md#gpuprofiler)** - GPU profiling context manager
- **[profile()](utils/profiler.md#profile)** - Profiling decorator
- **[MemoryTracker](utils/profiler.md#memorytracker)** - Memory usage tracker
- **[benchmark()](utils/profiler.md#benchmark)** - Function benchmarking

## Bridges

### NumPy Bridge

- **[NumpyBridge](bridges/numpy.md#numpybridge)** - NumPy acceleration bridge

### SciPy Bridge

- **[ScipyBridge](bridges/scipy.md#scipybridge)** - SciPy acceleration bridge

### Uproot Bridge

- **[UprootBridge](bridges/uproot.md#uprootbridge)** - Uproot/Awkward acceleration bridge
- **[HEPAccelerator](bridges/uproot.md#hepaccelerator)** - HEP helper class

---

*Note: This is a placeholder for auto-generated API documentation. Use tools like Sphinx or pdoc to generate comprehensive API docs from docstrings.*

## Generating API Documentation

### Using Sphinx

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Generate documentation
cd docs/api
sphinx-quickstart
sphinx-apidoc -o . ../../cudacc
make html
```

### Using pdoc

```bash
# Install pdoc
pip install pdoc3

# Generate documentation
pdoc --html --output-dir docs/api cudacc
```

### Using mkdocs

```bash
# Install mkdocs with plugins
pip install mkdocs mkdocstrings mkdocs-material

# Configure mkdocs.yml
# Build documentation
mkdocs build
```
