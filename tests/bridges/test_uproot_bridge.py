"""
Tests for uproot bridge.
"""

import pytest
import numpy as np


class TestUprootBridge:
    """Test suite for uproot bridge."""
    
    def test_uproot_bridge_supports_uproot(self):
        """Test that bridge recognizes uproot."""
        from cudacc.bridges.uproot_bridge import UprootBridge
        
        try:
            import uproot
            bridge = UprootBridge()
            assert bridge.supports(uproot)
        except ImportError:
            pytest.skip("uproot not installed")
    
    def test_uproot_bridge_supports_awkward(self):
        """Test that bridge recognizes awkward."""
        from cudacc.bridges.uproot_bridge import UprootBridge
        
        try:
            import awkward
            bridge = UprootBridge()
            assert bridge.supports(awkward)
        except ImportError:
            pytest.skip("awkward not installed")
    
    @pytest.mark.cuda
    def test_uproot_bridge_apply(self, skip_if_no_cuda):
        """Test applying the uproot bridge."""
        from cudacc.bridges.uproot_bridge import UprootBridge
        
        try:
            import uproot
            
            bridge = UprootBridge()
            uproot_gpu = bridge.apply(uproot, devices=[0])
            
            assert uproot_gpu is not None
            assert hasattr(uproot_gpu, '_cudacc_devices')
            assert hasattr(uproot_gpu, 'gpu_invariant_mass')
        except ImportError:
            pytest.skip("uproot or CuPy not installed")
    
    @pytest.mark.cuda
    def test_hep_accelerator_class(self, skip_if_no_cuda):
        """Test HEPAccelerator helper class."""
        from cudacc.bridges.uproot_bridge import UprootBridge
        
        try:
            import uproot
            
            bridge = UprootBridge()
            uproot_gpu = bridge.apply(uproot, devices=[0])
            
            assert hasattr(uproot_gpu, 'HEPAccelerator')
            assert hasattr(uproot_gpu.HEPAccelerator, 'compute_invariant_mass')
            assert hasattr(uproot_gpu.HEPAccelerator, 'filter_by_pt')
        except ImportError:
            pytest.skip("uproot or CuPy not installed")
    
    @pytest.mark.cuda
    def test_gpu_invariant_mass_available(self, skip_if_no_cuda):
        """Test that GPU invariant mass function is available."""
        from cudacc.bridges.uproot_bridge import UprootBridge
        
        try:
            import uproot
            
            bridge = UprootBridge()
            uproot_gpu = bridge.apply(uproot, devices=[0])
            
            # Create simple test data
            e1 = np.array([1.0], dtype=np.float32)
            px1 = np.array([0.0], dtype=np.float32)
            py1 = np.array([0.0], dtype=np.float32)
            pz1 = np.array([0.0], dtype=np.float32)
            
            e2 = np.array([1.0], dtype=np.float32)
            px2 = np.array([0.0], dtype=np.float32)
            py2 = np.array([0.0], dtype=np.float32)
            pz2 = np.array([0.0], dtype=np.float32)
            
            result = uproot_gpu.gpu_invariant_mass(
                e1, px1, py1, pz1, e2, px2, py2, pz2
            )
            
            assert result is not None
        except ImportError:
            pytest.skip("uproot or CuPy not installed")
