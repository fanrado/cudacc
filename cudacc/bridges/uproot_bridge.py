"""
Uproot bridge: HEP-specific acceleration (uproot/awkward).

Provides GPU acceleration for High Energy Physics data analysis workflows.
"""

from typing import Any
from ..registry import PackageBridge
from ..kernels.physics import gpu_invariant_mass, gpu_particle_distance
import types


class UprootBridge(PackageBridge):
    """
    Bridge for accelerating uproot/awkward array operations.
    
    This bridge focuses on HEP-specific operations like:
    - Invariant mass calculations
    - Lorentz transformations
    - Particle selection and filtering
    """
    
    def supports(self, pkg: Any) -> bool:
        """Check if this is uproot or awkward."""
        if not hasattr(pkg, '__name__'):
            return False
        return pkg.__name__ in ('uproot', 'awkward')
    
    def apply(self, pkg: Any, devices: list[int]) -> Any:
        """
        Create an accelerated version with HEP-specific GPU kernels.
        
        Args:
            pkg: The uproot or awkward module.
            devices: List of GPU device IDs.
        
        Returns:
            Accelerated module with GPU kernels.
        """
        try:
            import cupy as cp
        except ImportError:
            raise RuntimeError("CuPy is required for uproot/awkward acceleration")
        
        # Create wrapper module
        accelerated = types.ModuleType(f'{pkg.__name__}_accelerated')
        accelerated.__doc__ = f"GPU-accelerated {pkg.__name__} via cudacc"
        
        # Set default device
        if devices:
            cp.cuda.Device(devices[0]).use()
        
        # Copy original module attributes
        for attr_name in dir(pkg):
            if not attr_name.startswith('_'):
                setattr(accelerated, attr_name, getattr(pkg, attr_name))
        
        # Add HEP-specific GPU functions
        accelerated.gpu_invariant_mass = gpu_invariant_mass
        accelerated.gpu_particle_distance = gpu_particle_distance
        
        # Create helper class for common HEP operations
        class HEPAccelerator:
            """Helper class for HEP operations on GPU."""
            
            @staticmethod
            def compute_invariant_mass(events, particle1_idx, particle2_idx):
                """
                Compute invariant mass for particle pairs.
                
                Args:
                    events: Awkward array of events.
                    particle1_idx: Index of first particle in each event.
                    particle2_idx: Index of second particle in each event.
                
                Returns:
                    Array of invariant masses.
                """
                # Extract four-momentum components
                # This is a placeholder - actual implementation depends on event structure
                e1 = events['E'][particle1_idx]
                px1 = events['px'][particle1_idx]
                py1 = events['py'][particle1_idx]
                pz1 = events['pz'][particle1_idx]
                
                e2 = events['E'][particle2_idx]
                px2 = events['px'][particle2_idx]
                py2 = events['py'][particle2_idx]
                pz2 = events['pz'][particle2_idx]
                
                return gpu_invariant_mass(e1, px1, py1, pz1, e2, px2, py2, pz2)
            
            @staticmethod
            def filter_by_pt(events, pt_min, pt_max=None):
                """
                Filter particles by transverse momentum on GPU.
                
                Args:
                    events: Awkward array of events.
                    pt_min: Minimum pT.
                    pt_max: Maximum pT (optional).
                
                Returns:
                    Filtered events.
                """
                try:
                    import awkward as ak
                    
                    # Extract pt
                    pt = ak.to_cupy(events['pt'])
                    
                    # Apply filter
                    mask = pt >= pt_min
                    if pt_max is not None:
                        mask = mask & (pt <= pt_max)
                    
                    return events[ak.from_cupy(mask)]
                except Exception as e:
                    raise RuntimeError(f"GPU filtering failed: {e}")
        
        accelerated.HEPAccelerator = HEPAccelerator
        
        # Store device info
        accelerated._cudacc_devices = devices
        accelerated._cudacc_backend = 'numba+awkward'
        
        return accelerated


# Register this bridge
from ..registry import register_bridge
register_bridge('uproot', UprootBridge)
register_bridge('awkward', UprootBridge)
