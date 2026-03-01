"""
Tests for physics kernels.
"""

import pytest
import numpy as np


class TestPhysicsKernels:
    """Test suite for physics kernels."""
    
    @pytest.mark.cuda
    def test_gpu_particle_distance(self, skip_if_no_cuda, cleanup_gpu):
        """Test particle distance calculation."""
        from cudacc.kernels.physics import gpu_particle_distance
        
        # Create simple test case
        x1 = np.array([0, 0, 0], dtype=np.float32)
        y1 = np.array([0, 0, 0], dtype=np.float32)
        z1 = np.array([0, 0, 0], dtype=np.float32)
        
        x2 = np.array([3, 0, 0], dtype=np.float32)
        y2 = np.array([0, 4, 0], dtype=np.float32)
        z2 = np.array([0, 0, 5], dtype=np.float32)
        
        result = gpu_particle_distance(x1, y1, z1, x2, y2, z2)
        
        # Expected: [3, 4, 5]
        expected = np.array([3, 4, 5], dtype=np.float32)
        
        import cupy as cp
        result_cpu = cp.asnumpy(result)
        
        assert np.allclose(result_cpu, expected, rtol=1e-5)
    
    @pytest.mark.cuda
    def test_gpu_invariant_mass(self, skip_if_no_cuda, cleanup_gpu):
        """Test invariant mass calculation."""
        from cudacc.kernels.physics import gpu_invariant_mass
        
        # Create simple test case: two particles at rest with mass 1
        # E = m, p = 0 for each
        e1 = np.array([1.0], dtype=np.float32)
        px1 = np.array([0.0], dtype=np.float32)
        py1 = np.array([0.0], dtype=np.float32)
        pz1 = np.array([0.0], dtype=np.float32)
        
        e2 = np.array([1.0], dtype=np.float32)
        px2 = np.array([0.0], dtype=np.float32)
        py2 = np.array([0.0], dtype=np.float32)
        pz2 = np.array([0.0], dtype=np.float32)
        
        result = gpu_invariant_mass(e1, px1, py1, pz1, e2, px2, py2, pz2)
        
        # Expected: sqrt((1+1)^2 - 0) = 2
        expected = np.array([2.0], dtype=np.float32)
        
        import cupy as cp
        result_cpu = cp.asnumpy(result)
        
        assert np.allclose(result_cpu, expected, rtol=1e-4)
    
    @pytest.mark.cuda
    @pytest.mark.slow
    def test_invariant_mass_realistic(self, skip_if_no_cuda, cleanup_gpu):
        """Test invariant mass with realistic particle data."""
        from cudacc.kernels.physics import gpu_invariant_mass
        
        # Simulate some particles
        n = 1000
        
        # Random four-momenta
        e1 = np.random.uniform(1, 10, n).astype(np.float32)
        px1 = np.random.randn(n).astype(np.float32)
        py1 = np.random.randn(n).astype(np.float32)
        pz1 = np.random.randn(n).astype(np.float32)
        
        e2 = np.random.uniform(1, 10, n).astype(np.float32)
        px2 = np.random.randn(n).astype(np.float32)
        py2 = np.random.randn(n).astype(np.float32)
        pz2 = np.random.randn(n).astype(np.float32)
        
        result = gpu_invariant_mass(e1, px1, py1, pz1, e2, px2, py2, pz2)
        
        # Compute expected on CPU
        e_tot = e1 + e2
        px_tot = px1 + px2
        py_tot = py1 + py2
        pz_tot = pz1 + pz2
        p_squared = px_tot**2 + py_tot**2 + pz_tot**2
        m_squared = e_tot**2 - p_squared
        expected = np.sqrt(np.abs(m_squared))
        
        import cupy as cp
        result_cpu = cp.asnumpy(result)
        
        assert np.allclose(result_cpu, expected, rtol=1e-4)
