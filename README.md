# CUDACC

> Transparent CUDA acceleration for scientific Python packages

**cudacc** is a Python library that provides transparent GPU acceleration for popular scientific computing packages like NumPy, SciPy, and domain-specific tools.

## Overview

cudacc works by intercepting operations from CPU-based libraries and dispatching them to optimized GPU kernels implemented with Numba CUDA and CuPy.

## Key Features

- ✨ **Transparent Acceleration**: Use familiar NumPy/SciPy syntax with automatic GPU execution
- ⚡ **Custom Kernels**: Hand-optimized Numba CUDA kernels for common operations
- 🖥️ **Multi-GPU Support**: Automatic device detection and multi-GPU coordination
- 🔬 **Domain-Specific Bridges**: Specialized support for HEP (High Energy Physics) and CFD (Computational Fluid DYnamics) workflows
- 💾 **Memory Management**: Efficient GPU memory pooling and management
- 📊 **Profiling Tools**: Built-in timing and memory profiling utilities

## Supported Packages

- **NumPy**: Array operations, reductions, transformations
- **SciPy**: FFT, linear algebra, signal processing
- **uproot/awkward**: HEP-specific data analysis (ROOT files, jagged arrays)

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with CUDA Compute Capability 5.0 or higher
- Recommended: 8GB+ GPU memory for large datasets

### Software Requirements

- Linux, Windows, or macOS
- Python 3.8 or higher
- NVIDIA GPU drivers (check with `nvidia-smi`)
- CUDA 11.0+ or CUDA 12.x

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cudacc.git
cd cudacc
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv env_cudacc

# Activate (Linux/macOS)
source env_cudacc/bin/activate

# Activate (Windows)
env_cudacc\Scripts\activate
```

### 3. Install CUDA Dependencies

The package requires CuPy and CUDA support. Use the provided setup script:

```bash
# Run the CUDA setup script
./setup_cuda.sh
```

This script will:

- Detect your NVIDIA GPU and CUDA version
- Install the appropriate CuPy package (cupy-cuda12x or cupy-cuda11x)
- Verify CUDA is working correctly
- Run a simple GPU operation test

**Manual CuPy Installation (if needed):**

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

### 4. Install cudacc

```bash
# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e .[dev]
```

### 5. Verify Installation

```bash
python -c "import cupy; print(f'CuPy: {cupy.__version__}')"
python -c "from cudacc import accelerate; print('cudacc imported successfully')"
```

## Quick Start

### Basic NumPy Acceleration

```python
import numpy as np
from cudacc import accelerate

# Accelerate NumPy for GPU 0
np_gpu = accelerate(np, devices=[0])

# Create data on CPU, transfer to GPU
data_cpu = np.random.randn(10_000_000).astype(np.float32)
data_gpu = np_gpu.array(data_cpu)

# Perform operations on GPU
result = np_gpu.sum(data_gpu)
mean = np_gpu.mean(data_gpu)
std = np_gpu.std(data_gpu)

print(f"Sum: {result}, Mean: {mean}, Std: {std}")
```

### Auto Device Selection

```python
from cudacc import accelerate
import numpy as np

# Auto-detect and use all available GPUs
np_gpu = accelerate(np)

# Create and transfer data
data_cpu = np.random.randn(1_000_000).astype(np.float32)
data_gpu = np_gpu.array(data_cpu)

# Chain operations (all on GPU)
normalized = (data_gpu - np_gpu.mean(data_gpu)) / np_gpu.std(data_gpu)
result = np_gpu.sum(normalized ** 2)
```

### HEP Analysis Example

```python
import uproot
from cudacc import accelerate

# Accelerate uproot for particle physics analysis
uproot_gpu = accelerate(uproot)

# Compute invariant mass on GPU
masses = uproot_gpu.gpu_invariant_mass(e1, px1, py1, pz1, e2, px2, py2, pz2)
```

### Device Management

```python
from cudacc.utils.device import print_device_info, detect_devices

# List all available CUDA devices
devices = detect_devices()
print(f"Found {len(devices)} CUDA device(s)")

# Print detailed device information
print_device_info()
```

## Testing

The package includes comprehensive test suites for all components. Use the provided test script:

```bash
# Run all tests with detailed output
./run_tests.sh
```

This will:

- Check your Python and CUDA environment
- Run all component tests with maximum verbosity
- Generate detailed logs in `output_test/` directory
- Show which tests pass, fail, or are skipped
- Explain why tests are skipped (e.g., missing CUDA, missing dependencies)

For more details, see [TESTING_README.md](TESTING_README.md)

## Project Structure

```
cudacc/
├── cudacc/              # Main package
│   ├── accelerator.py   # Main acceleration API
│   ├── dispatcher.py    # Kernel routing and management
│   ├── memory.py        # GPU memory management
│   ├── registry.py      # Package registration
│   ├── bridges/         # Package-specific bridges
│   │   ├── numpy_bridge.py
│   │   ├── scipy_bridge.py
│   │   └── uproot_bridge.py
│   ├── kernels/         # GPU kernel implementations
│   │   ├── physics.py
│   │   ├── reductions.py
│   │   └── transforms.py
│   └── utils/           # Utilities
│       ├── device.py
│       └── profiler.py
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Usage examples
├── setup_cuda.sh        # CUDA setup script
├── run_tests.sh         # Test runner script
└── README.md            # This file
```

## Architecture

cudacc uses a **bridge pattern** to support different packages:

1. **Registry**: Maps packages to their acceleration bridges
2. **Bridges**: Package-specific interception logic
3. **Dispatcher**: Routes operations to appropriate GPU kernels
4. **Kernels**: Numba CUDA implementations of operations
5. **Memory Manager**: Handles GPU memory allocation and pooling

## Examples

See the [examples/](examples/) directory for more usage examples:

- `particle_physics/basic_acceleration.py` - Basic NumPy GPU acceleration
- `particle_physics/uproot_demo.py` - HEP analysis with uproot
- `fluid_dynamics/fenics_demo.py` - CFD simulation acceleration

## Documentation

For detailed documentation, see the [docs/](docs/) directory:

- [Installation Guide](docs/installation.md)
- [Quick Start Guide](docs/quickstart.md)
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api/)

## Troubleshooting

### "No module named 'cupy'"

Run `./setup_cuda.sh` or manually install CuPy for your CUDA version.

### "RuntimeError: No CUDA devices available"

- Check if your GPU is recognized: `nvidia-smi`
- Verify CuPy can detect CUDA: `python -c "import cupy; print(cupy.cuda.is_available())"`
- Ensure NVIDIA drivers are properly installed

### Tests are skipped

Most tests require CUDA hardware to run. Check detailed logs in `output_test/` to see why specific tests were skipped.

## Contributing

This project is under active development. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite (`./run_tests.sh`)
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub.
