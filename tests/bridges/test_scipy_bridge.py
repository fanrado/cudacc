"""
Tests for SciPy bridge.
"""

import pytest


class TestScipyBridge:
    """Test suite for SciPy bridge."""
    
    def test_scipy_bridge_supports(self):
        """Test that bridge recognizes SciPy."""
        from cudacc.bridges.scipy_bridge import ScipyBridge
        
        try:
            import scipy
            bridge = ScipyBridge()
            assert bridge.supports(scipy)
        except ImportError:
            pytest.skip("SciPy not installed")
    
    @pytest.mark.cuda
    def test_scipy_bridge_apply(self, skip_if_no_cuda):
        """Test applying the SciPy bridge."""
        from cudacc.bridges.scipy_bridge import ScipyBridge
        
        try:
            import scipy
            
            bridge = ScipyBridge()
            scipy_gpu = bridge.apply(scipy, devices=[0])
            
            assert scipy_gpu is not None
            assert hasattr(scipy_gpu, '_cudacc_devices')
        except ImportError:
            pytest.skip("SciPy or CuPy not installed")
    
    @pytest.mark.cuda
    def test_scipy_fft(self, skip_if_no_cuda, cleanup_gpu):
        """Test SciPy FFT acceleration."""
        from cudacc.bridges.scipy_bridge import ScipyBridge
        import numpy as np
        
        try:
            import scipy
            
            bridge = ScipyBridge()
            scipy_gpu = bridge.apply(scipy, devices=[0])
            
            # Test FFT
            if hasattr(scipy_gpu, 'fft'):
                import cupy as cp
                data = cp.random.randn(100).astype(cp.float32)
                result = scipy_gpu.fft.fft(data)
                
                assert result is not None
            else:
                pytest.skip("FFT not available in accelerated SciPy")
        except ImportError:
            pytest.skip("SciPy or CuPy not installed")
