# Installation Guide

This guide covers the installation of cudacc and its dependencies.

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with CUDA Compute Capability 5.0 or higher
- Recommended: 8GB+ GPU memory for large datasets

### Software Requirements

- Linux, Windows, or macOS
- Python 3.8 or higher
- NVIDIA CUDA Toolkit 11.0 or higher
- NVIDIA GPU drivers (compatible with your CUDA version)

## Step 1: Install CUDA Toolkit

### Linux

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Windows

Download and install from: https://developer.nvidia.com/cuda-downloads

### macOS

CUDA is not officially supported on macOS since macOS 10.14. Consider using Docker or Linux.

## Step 2: Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

You should see your CUDA version and GPU information.

## Step 3: Create Virtual Environment (Recommended)

```bash
python -m venv cudacc-env
source cudacc-env/bin/activate  # On Windows: cudacc-env\Scripts\activate
```

## Step 4: Install CuPy

CuPy is the core dependency for GPU array operations.

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# Verify installation
python -c "import cupy; print(cupy.__version__)"
```

## Step 5: Install Numba

```bash
pip install numba

# Verify CUDA support
python -c "from numba import cuda; print(cuda.is_available())"
```

## Step 6: Install cudacc

### From PyPI (Recommended)

```bash
pip install cudacc
```

### From Source

```bash
git clone https://github.com/yourusername/cudacc.git
cd cudacc
pip install -e .
```

## Step 7: Install Optional Dependencies

### For HEP Support

```bash
pip install uproot awkward
```

### For SciPy Acceleration

CuPy includes scipy support by default, but verify:

```bash
python -c "import cupyx.scipy; print('SciPy support available')"
```

### For Memory Management (RMM)

```bash
# Install RAPIDS Memory Manager for advanced memory pooling
pip install rmm-cu11  # For CUDA 11.x
```

## Verification

Create a test script `test_install.py`:

```python
#!/usr/bin/env python
"""Test cudacc installation."""

import sys

def test_cuda():
    """Test CUDA availability."""
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"✓ CUDA available with {device_count} device(s)")
        
        # Get device properties
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  Device 0: {props['name'].decode('utf-8')}")
        return True
    except Exception as e:
        print(f"✗ CUDA not available: {e}")
        return False

def test_numba():
    """Test Numba CUDA support."""
    try:
        from numba import cuda
        if cuda.is_available():
            print("✓ Numba CUDA support available")
            return True
        else:
            print("✗ Numba CUDA support not available")
            return False
    except Exception as e:
        print(f"✗ Numba error: {e}")
        return False

def test_cudacc():
    """Test cudacc installation."""
    try:
        import cudacc
        print(f"✓ cudacc installed (version {cudacc.__version__})")
        return True
    except Exception as e:
        print(f"✗ cudacc not available: {e}")
        return False

def test_acceleration():
    """Test basic acceleration."""
    try:
        import numpy as np
        from cudacc import accelerate
        
        np_gpu = accelerate(np, devices=[0])
        arr = np_gpu.array([1, 2, 3, 4, 5])
        result = np_gpu.sum(arr)
        
        if abs(result - 15) < 1e-5:
            print("✓ Basic acceleration working")
            return True
        else:
            print("✗ Acceleration test failed")
            return False
    except Exception as e:
        print(f"✗ Acceleration test error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("cudacc Installation Verification")
    print("=" * 60)
    print()
    
    results = []
    results.append(("CUDA", test_cuda()))
    results.append(("Numba", test_numba()))
    results.append(("cudacc", test_cudacc()))
    results.append(("Acceleration", test_acceleration()))
    
    print()
    print("=" * 60)
    
    if all(r[1] for r in results):
        print("✓ All tests passed! cudacc is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

Run the test:

```bash
python test_install.py
```

## Troubleshooting

### "CUDA not available"

- Verify CUDA toolkit installation: `nvcc --version`
- Check GPU driver: `nvidia-smi`
- Ensure CuPy is installed for your CUDA version

### "Numba CUDA not available"

- Reinstall Numba: `pip uninstall numba && pip install numba`
- Check CUDA_PATH environment variable

### Import errors

- Verify all dependencies are installed in the same environment
- Check Python version compatibility (3.8+)

### Performance issues

- Update GPU drivers to the latest version
- Ensure sufficient GPU memory
- Check for thermal throttling with `nvidia-smi`

## Docker Installation (Alternative)

For a containerized environment:

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install cupy-cuda11x numba cudacc

# Verify installation
RUN python3 -c "import cudacc; print(cudacc.__version__)"
```

Build and run:

```bash
docker build -t cudacc .
docker run --gpus all -it cudacc
```

## Next Steps

- Read the [Quick Start Guide](quickstart.md)
- Explore [Examples](../examples/)
- Check the [Architecture](architecture.md) documentation
