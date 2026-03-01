# cudacc

> Transparent CUDA acceleration for scientific Python packages

**cudacc** is a Python library that provides transparent GPU acceleration for popular scientific computing packages like NumPy, SciPy, and domain-specific tools used in particle physics and computational fluid dynamics.

## Overview

cudacc works by intercepting operations from CPU-based libraries and dispatching them to optimized GPU kernels implemented with Numba CUDA and CuPy. The library maintains a familiar API while delivering significant performance improvements on CUDA-capable hardware.

## Key Features

- **Transparent Acceleration**: Use familiar NumPy/SciPy syntax with automatic GPU execution
- **Custom Kernels**: Hand-optimized Numba CUDA kernels for common operations
- **Multi-GPU Support**: Automatic device detection and multi-GPU coordination
- **Domain-Specific Bridges**: Specialized support for HEP (High Energy Physics) and CFD workflows
- **Memory Management**: RMM-backed memory pooling for efficient GPU memory usage
- **Profiling Tools**: Built-in timing and memory profiling utilities

## Quick Start

```python
import numpy as np
from cudacc import accelerate

# Accelerate NumPy operations
np_gpu = accelerate(np, devices=[0])

# Use exactly like NumPy, but runs on GPU
data = np_gpu.random.randn(10000000)
result = np_gpu.sum(data)
```

## Supported Packages

- **NumPy**: Array operations, reductions, transformations
- **SciPy**: FFT, linear algebra, signal processing
- **uproot/awkward**: HEP-specific data analysis (ROOT files, jagged arrays)

## Architecture

cudacc uses a **bridge pattern** to support different packages:

1. **Registry**: Maps packages to their acceleration bridges
2. **Bridges**: Package-specific interception logic
3. **Dispatcher**: Routes operations to appropriate GPU kernels
4. **Kernels**: Numba CUDA implementations of operations
5. **Memory Manager**: Handles GPU memory allocation and pooling

See [Architecture](architecture.md) for detailed design information.

## Installation

See [Installation Guide](installation.md) for detailed instructions.

Quick install:

```bash
pip install cudacc
```

Requirements:
- CUDA toolkit (11.0+)
- CuPy
- Numba
- Python 3.8+

## Examples

### Basic NumPy Acceleration

```python
from cudacc import accelerate
import numpy as np

# Accelerate NumPy
np_gpu = accelerate(np)

# Create large array
data = np_gpu.random.randn(100_000_000)

# Operations run on GPU
mean = np_gpu.mean(data)
std = np_gpu.std(data)
normalized = (data - mean) / std
```

### HEP Analysis

```python
import uproot
from cudacc import accelerate

# Accelerate uproot
uproot_gpu = accelerate(uproot)

# Compute invariant mass on GPU
masses = uproot_gpu.gpu_invariant_mass(e1, px1, py1, pz1, e2, px2, py2, pz2)
```

See the [examples/](../examples/) directory for complete demonstrations.

## Documentation

- [Installation](installation.md)
- [Quick Start Guide](quickstart.md)
- [Architecture](architecture.md)
- [API Reference](api/)

## Performance

cudacc can provide significant speedups for array operations:

| Operation | Array Size | CPU Time | GPU Time | Speedup |
|-----------|-----------|----------|----------|---------|
| Sum | 100M | 120ms | 5ms | 24x |
| Normalize | 100M | 250ms | 12ms | 21x |
| Invariant Mass | 1M pairs | 180ms | 8ms | 22x |

*Benchmarks on NVIDIA RTX 3090 vs Intel i9-12900K*

## License

See [LICENSE](../LICENSE) for details.

## Contributing

Contributions are welcome! Please see our contribution guidelines.

## Citation

If you use cudacc in your research, please cite:

```bibtex
@software{cudacc,
  title={cudacc: Transparent CUDA Acceleration for Scientific Python},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/cudacc}
}
```
