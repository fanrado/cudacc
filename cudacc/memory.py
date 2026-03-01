"""
GPU memory management with RMM backend.

Provides memory pool management, allocation tracking, and
multi-GPU memory coordination.
"""

from typing import Optional, List
import numpy as np


try:
    import cupy as cp
    import rmm
    HAS_RMM = True
except ImportError:
    HAS_RMM = False


class MemoryPool:
    """
    Manages GPU memory allocation and pooling.
    
    Uses RMM (RAPIDS Memory Manager) when available for efficient
    memory reuse and allocation.
    """
    
    def __init__(self, device_id: int, pool_size: Optional[int] = None):
        """
        Initialize memory pool for a specific device.
        
        Args:
            device_id: CUDA device ID.
            pool_size: Initial pool size in bytes. If None, uses default.
        """
        self.device_id = device_id
        self.pool_size = pool_size
        self._initialized = False
        
        if HAS_RMM:
            self._init_rmm()
        else:
            self._init_cupy()
    
    def _init_rmm(self):
        """Initialize RMM memory pool."""
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=self.pool_size,
            devices=[self.device_id]
        )
        self._initialized = True
    
    def _init_cupy(self):
        """Fallback to CuPy memory pool."""
        if 'cp' in globals():
            with cp.cuda.Device(self.device_id):
                pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(pool.malloc)
                if self.pool_size:
                    pool.set_limit(size=self.pool_size)
            self._initialized = True
    
    def allocate(self, size: int, dtype=np.float32):
        """
        Allocate GPU memory.
        
        Args:
            size: Number of elements.
            dtype: Data type.
        
        Returns:
            GPU array of the requested size.
        """
        if not self._initialized:
            raise RuntimeError("Memory pool not initialized")
        
        if 'cp' in globals():
            with cp.cuda.Device(self.device_id):
                return cp.empty(size, dtype=dtype)
        else:
            raise RuntimeError("CuPy not available")
    
    def free_unused(self):
        """Free unused memory from the pool."""
        if 'cp' in globals():
            with cp.cuda.Device(self.device_id):
                cp.get_default_memory_pool().free_all_blocks()


class MultiGPUMemoryManager:
    """
    Coordinates memory across multiple GPUs.
    """
    
    def __init__(self, device_ids: List[int]):
        """
        Initialize manager for multiple devices.
        
        Args:
            device_ids: List of CUDA device IDs to manage.
        """
        self.device_ids = device_ids
        self.pools = {
            device_id: MemoryPool(device_id)
            for device_id in device_ids
        }
    
    def get_pool(self, device_id: int) -> MemoryPool:
        """Get the memory pool for a specific device."""
        return self.pools[device_id]
    
    def free_all_unused(self):
        """Free unused memory on all devices."""
        for pool in self.pools.values():
            pool.free_unused()
