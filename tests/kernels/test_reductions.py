"""
Tests for reduction kernels.
"""

import pytest
import numpy as np


class TestReductionKernels:
    """Test suite for reduction kernels."""
    
    @pytest.mark.cuda
    def test_gpu_sum(self, skip_if_no_cuda, sample_array, cleanup_gpu):
        """Test GPU sum reduction."""
        from cudacc.kernels.reductions import gpu_sum
        
        result = gpu_sum(sample_array)
        expected = np.sum(sample_array)
        
        assert np.isclose(result, expected, rtol=1e-4)
    
    @pytest.mark.cuda
    def test_gpu_min(self, skip_if_no_cuda, sample_array, cleanup_gpu):
        """Test GPU min reduction."""
        from cudacc.kernels.reductions import gpu_min
        
        result = gpu_min(sample_array)
        expected = np.min(sample_array)
        
        assert np.isclose(result, expected, rtol=1e-4)
    
    @pytest.mark.cuda
    def test_gpu_max(self, skip_if_no_cuda, sample_array, cleanup_gpu):
        """Test GPU max reduction."""
        from cudacc.kernels.reductions import gpu_max
        
        result = gpu_max(sample_array)
        expected = np.max(sample_array)
        
        assert np.isclose(result, expected, rtol=1e-4)
    
    @pytest.mark.cuda
    def test_gpu_histogram(self, skip_if_no_cuda, cleanup_gpu):
        """Test GPU histogram."""
        from cudacc.kernels.reductions import gpu_histogram
        
        data = np.random.randn(10000).astype(np.float32)
        bins = np.linspace(-3, 3, 11).astype(np.float32)
        
        result = gpu_histogram(data, bins)
        expected, _ = np.histogram(data, bins)
        
        # Allow some difference due to floating point
        assert np.allclose(result, expected, rtol=0.1)
    
    @pytest.mark.cuda
    def test_sum_empty_array(self, skip_if_no_cuda, cleanup_gpu):
        """Test sum on empty array."""
        from cudacc.kernels.reductions import gpu_sum
        
        empty = np.array([], dtype=np.float32)
        result = gpu_sum(empty)
        
        assert result == 0.0
