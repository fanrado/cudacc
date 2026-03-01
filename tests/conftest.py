"""
pytest configuration and fixtures for cudacc tests.

Provides common fixtures and CUDA availability checks.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA device"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        return device_count > 0
    except Exception:
        return False


@pytest.fixture(scope="session")
def skip_if_no_cuda(cuda_available):
    """Skip test if CUDA is not available."""
    if not cuda_available:
        pytest.skip("CUDA not available")


@pytest.fixture
def gpu_device():
    """Provide a GPU device for testing."""
    try:
        import cupy as cp
        return cp.cuda.Device(0)
    except Exception:
        pytest.skip("No GPU device available")


@pytest.fixture
def sample_array():
    """Provide a sample NumPy array for testing."""
    import numpy as np
    return np.random.randn(1000).astype(np.float32)


@pytest.fixture
def sample_2d_array():
    """Provide a sample 2D NumPy array for testing."""
    import numpy as np
    return np.random.randn(100, 100).astype(np.float32)


@pytest.fixture
def cleanup_gpu():
    """Clean up GPU memory after test."""
    yield
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
