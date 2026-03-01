"""
Tests for transform kernels.
"""

import pytest
import numpy as np


class TestTransformKernels:
    """Test suite for transform kernels."""
    
    @pytest.mark.cuda
    def test_gpu_multiply(self, skip_if_no_cuda, cleanup_gpu):
        """Test GPU element-wise multiplication."""
        from cudacc.kernels.transforms import gpu_multiply
        
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(100).astype(np.float32)
        
        result = gpu_multiply(a, b)
        expected = a * b
        
        import cupy as cp
        result_cpu = cp.asnumpy(result)
        
        assert np.allclose(result_cpu, expected, rtol=1e-5)
    
    @pytest.mark.cuda
    def test_gpu_add(self, skip_if_no_cuda, cleanup_gpu):
        """Test GPU element-wise addition."""
        from cudacc.kernels.transforms import gpu_add
        
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(100).astype(np.float32)
        
        result = gpu_add(a, b)
        expected = a + b
        
        import cupy as cp
        result_cpu = cp.asnumpy(result)
        
        assert np.allclose(result_cpu, expected, rtol=1e-5)
    
    @pytest.mark.cuda
    def test_gpu_normalize(self, skip_if_no_cuda, sample_array, cleanup_gpu):
        """Test GPU normalization."""
        from cudacc.kernels.transforms import gpu_normalize
        
        result = gpu_normalize(sample_array)
        
        # Compute expected normalization
        mean = np.mean(sample_array)
        std = np.std(sample_array)
        expected = (sample_array - mean) / std
        
        import cupy as cp
        result_cpu = cp.asnumpy(result)
        
        assert np.allclose(result_cpu, expected, rtol=1e-5)
    
    @pytest.mark.cuda
    def test_gpu_clip(self, skip_if_no_cuda, sample_array, cleanup_gpu):
        """Test GPU clipping."""
        from cudacc.kernels.transforms import gpu_clip
        
        min_val = -1.0
        max_val = 1.0
        
        result = gpu_clip(sample_array, min_val, max_val)
        expected = np.clip(sample_array, min_val, max_val)
        
        import cupy as cp
        result_cpu = cp.asnumpy(result)
        
        assert np.allclose(result_cpu, expected, rtol=1e-5)
    
    @pytest.mark.cuda
    def test_normalize_with_provided_stats(self, skip_if_no_cuda, cleanup_gpu):
        """Test normalization with provided mean and std."""
        from cudacc.kernels.transforms import gpu_normalize
        
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        result = gpu_normalize(data, mean=3.0, std=1.0)
        expected = (data - 3.0) / 1.0
        
        import cupy as cp
        result_cpu = cp.asnumpy(result)
        
        assert np.allclose(result_cpu, expected)
