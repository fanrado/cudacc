# CUDACC Testing Guide

This directory contains scripts to help you set up CUDA and run comprehensive tests on the CUDACC package.

## Prerequisites

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (check with `nvidia-smi`)
- Python environment activated

## Quick Start

### 1. Activate Your Python Environment

```bash
source /media/rado/RADO/WORKSPACE/cudacc/env_cudacc/bin/activate
```

### 2. Setup CUDA (Install CuPy)

Run the CUDA setup script to install CuPy and verify your CUDA environment:

```bash
./setup_cuda.sh
```

This script will:
- Detect your NVIDIA GPU and CUDA version
- Install the appropriate CuPy package (cupy-cuda12x for CUDA 12.x)
- Verify that CUDA is working correctly
- Run a simple GPU operation test

### 3. Run the Test Suite

Once CUDA is set up, run the comprehensive test suite:

```bash
./run_tests.sh
```

This script will:
- Check your Python and CUDA environment
- Run all component tests with detailed logging
- Generate comprehensive reports in the `output_test/` directory
- Show which tests pass, fail, or are skipped
- Explain why tests are skipped (e.g., missing CUDA, missing dependencies)

## Test Output Files

All test results are saved in the `output_test/` directory:

### Main Reports
- `test_report_latest.txt` - Symlink to the most recent test report
- `test_report_YYYYMMDD_HHMMSS.txt` - Timestamped comprehensive report
- `overall_summary.txt` - Summary of all tests run together
- `environment_info.txt` - Python and CUDA environment details

### Detailed Test Logs
- `detailed_test_accelerator.txt` - Core accelerator component tests
- `detailed_test_dispatcher.txt` - Kernel dispatcher tests
- `detailed_test_memory.txt` - Memory management tests
- `detailed_test_numpy_bridge.txt` - NumPy bridge tests
- `detailed_test_scipy_bridge.txt` - SciPy bridge tests
- `detailed_test_uproot_bridge.txt` - Uproot bridge tests
- `detailed_test_physics.txt` - Physics kernel tests
- `detailed_test_reductions.txt` - Reduction kernel tests
- `detailed_test_transforms.txt` - Transform kernel tests

Each detailed log contains:
- Full test output with maximum verbosity
- Local variable states when tests fail
- Complete stack traces for failures
- Reasons for skipped tests
- Line-by-line execution details

## Manual CuPy Installation

If the automatic setup doesn't work, you can install CuPy manually:

### For CUDA 12.x:
```bash
pip install cupy-cuda12x
```

### For CUDA 11.x:
```bash
pip install cupy-cuda11x
```

### Verify Installation:
```bash
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.getDeviceCount())"
```

## Troubleshooting

### "No module named 'cupy'"
- Run `./setup_cuda.sh` to install CuPy
- Or manually install with `pip install cupy-cuda12x`

### "CUDA device detection failed"
- Check if NVIDIA drivers are installed: `nvidia-smi`
- Verify CuPy can detect CUDA: `python -c "import cupy; print(cupy.cuda.is_available())"`

### Tests are skipped
- Most tests require CUDA to run
- Check the detailed logs to see why specific tests were skipped
- Common reasons: No CUDA devices, missing optional dependencies

### "RuntimeError: No CUDA devices available"
- Ensure your GPU is recognized by `nvidia-smi`
- Make sure CuPy is properly installed
- Check that your GPU has enough free memory

## Understanding Test Results

### PASSED ✓
- Test executed successfully
- All assertions passed
- Component is working correctly

### FAILED ✗
- Test ran but assertions failed
- Check the detailed log for the exact line and error
- Review stack traces and local variable values

### SKIPPED ⊘
- Test was not executed
- Usually due to:
  - Missing CUDA hardware/drivers
  - Missing optional dependencies (e.g., uproot)
  - Test marked to skip in certain conditions
- Check the skip reason in the detailed log

## Component Structure

### Core Components
- **Accelerator**: Main API for accelerating packages
- **Dispatcher**: Kernel routing and management
- **Memory**: GPU memory allocation and management

### Bridges
- **NumPy**: Acceleration for NumPy operations
- **SciPy**: Acceleration for SciPy scientific computing
- **Uproot**: Acceleration for particle physics ROOT file operations

### Kernels
- **Physics**: Physics-specific GPU kernels (kinematics, etc.)
- **Reductions**: Sum, mean, max, min operations
- **Transforms**: FFT, convolution, matrix operations

## Example: Running Individual Tests

You can also run individual test files manually:

```bash
# Run with maximum detail
python -m pytest tests/test_dispatcher.py -vvv --tb=long --showlocals

# Run specific test
python -m pytest tests/test_dispatcher.py::TestKernelDispatcher::test_dispatcher_creation -v

# Run all tests in a directory
python -m pytest tests/bridges/ -v
```

## Getting Help

If tests fail or you encounter issues:

1. Check `output_test/environment_info.txt` for your setup
2. Review the detailed test logs for specific failures
3. Look at the line numbers and local variables in failure tracebacks
4. Ensure all dependencies are installed: `pip install -e .[dev]`

## Additional Commands

### List all test files:
```bash
find tests -name "test_*.py"
```

### Check Python packages:
```bash
pip list | grep -E "cupy|numpy|scipy|uproot|pytest"
```

### Clean test outputs:
```bash
rm -rf output_test/
```

### Re-run only failed tests:
```bash
python -m pytest --lf -v
```
