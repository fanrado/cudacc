"""
Tests for the accelerate() entry point.
"""

import pytest
import numpy as np


class TestAccelerator:
    """Test suite for the main accelerate() function."""
    
    def test_accelerate_numpy(self, skip_if_no_cuda):
        """Test accelerating NumPy."""
        from cudacc import accelerate
        
        np_gpu = accelerate(np, devices=[0])
        
        # Check that it's accelerated
        assert hasattr(np_gpu, '_cudacc_devices')
        assert np_gpu._cudacc_devices == [0]
    
    def test_accelerate_no_devices(self, skip_if_no_cuda):
        """Test accelerate with auto device selection."""
        from cudacc import accelerate
        
        np_gpu = accelerate(np)
        
        # Should select available devices
        assert hasattr(np_gpu, '_cudacc_devices')
        assert len(np_gpu._cudacc_devices) > 0
    
    def test_accelerate_invalid_package(self):
        """Test accelerate with unsupported package."""
        from cudacc import accelerate
        import sys
        
        with pytest.raises(ValueError, match="not supported"):
            accelerate(sys)
    
    def test_accelerate_invalid_device(self, skip_if_no_cuda):
        """Test accelerate with invalid device ID."""
        from cudacc import accelerate
        
        with pytest.raises(ValueError):
            accelerate(np, devices=[999])
    
    def test_no_cuda_available(self, monkeypatch):
        """Test behavior when CUDA is not available."""
        from cudacc import accelerate
        # from cudacc.utils import device
        
        # # Mock detect_devices to return empty list
        # monkeypatch.setattr(device, 'detect_devices', lambda: [])
        monkeypatch.setattr('cudacc.utils.device.detect_devices', lambda: [])
        with pytest.raises(RuntimeError, match="No CUDA devices"):
            accelerate(np)
    
    def test_accelerated_operations(self, skip_if_no_cuda, sample_array):
        """Test that accelerated operations work."""
        from cudacc import accelerate
        
        np_gpu = accelerate(np, devices=[0])
        
        # Create array on GPU
        arr = np_gpu.array(sample_array)
        
        # Test basic operations
        result_sum = np_gpu.sum(arr)
        result_min = np_gpu.min(arr)
        result_max = np_gpu.max(arr)
        
        # Compare with CPU NumPy
        expected_sum = np.sum(sample_array)
        expected_min = np.min(sample_array)
        expected_max = np.max(sample_array)
        
        assert np.isclose(result_sum, expected_sum, rtol=1e-5)
        assert np.isclose(result_min, expected_min, rtol=1e-5)
        assert np.isclose(result_max, expected_max, rtol=1e-5)
