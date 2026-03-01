"""
Tests for NumPy bridge.
"""

import pytest
import numpy as np


class TestNumpyBridge:
    """Test suite for NumPy bridge."""
    
    def test_numpy_bridge_supports(self):
        """Test that bridge recognizes NumPy."""
        from cudacc.bridges.numpy_bridge import NumpyBridge
        
        bridge = NumpyBridge()
        assert bridge.supports(np)
    
    @pytest.mark.cuda
    def test_numpy_bridge_apply(self, skip_if_no_cuda):
        """Test applying the NumPy bridge."""
        from cudacc.bridges.numpy_bridge import NumpyBridge
        
        bridge = NumpyBridge()
        np_gpu = bridge.apply(np, devices=[0])
        
        assert np_gpu is not None
        assert hasattr(np_gpu, '_cudacc_devices')
        assert np_gpu._cudacc_devices == [0]
    
    @pytest.mark.cuda
    def test_accelerated_sum(self, skip_if_no_cuda, sample_array, cleanup_gpu):
        """Test that sum works on accelerated NumPy."""
        from cudacc.bridges.numpy_bridge import NumpyBridge
        
        bridge = NumpyBridge()
        np_gpu = bridge.apply(np, devices=[0])
        
        arr = np_gpu.array(sample_array)
        result = np_gpu.sum(arr)
        expected = np.sum(sample_array)
        
        assert np.isclose(result, expected, rtol=1e-5)
    
    @pytest.mark.cuda
    def test_accelerated_operations(self, skip_if_no_cuda, cleanup_gpu):
        """Test various operations on accelerated NumPy."""
        from cudacc.bridges.numpy_bridge import NumpyBridge
        
        bridge = NumpyBridge()
        np_gpu = bridge.apply(np, devices=[0])
        
        # Create arrays
        a = np_gpu.array([1, 2, 3, 4, 5], dtype=np.float32)
        b = np_gpu.array([2, 2, 2, 2, 2], dtype=np.float32)
        
        # Test operations
        result_add = np_gpu.add(a, b)
        result_mul = np_gpu.multiply(a, b)
        
        import cupy as cp
        expected_add = cp.array([3, 4, 5, 6, 7], dtype=np.float32)
        expected_mul = cp.array([2, 4, 6, 8, 10], dtype=np.float32)
        
        assert cp.allclose(result_add, expected_add)
        assert cp.allclose(result_mul, expected_mul)
