"""
Tests for GPU memory management.
"""

import pytest


class TestMemoryPool:
    """Test suite for MemoryPool."""
    
    @pytest.mark.cuda
    def test_memory_pool_creation(self, skip_if_no_cuda):
        """Test creating a memory pool."""
        from cudacc.memory import MemoryPool
        
        pool = MemoryPool(device_id=0)
        assert pool.device_id == 0
        assert pool._initialized
    
    @pytest.mark.cuda
    def test_memory_allocation(self, skip_if_no_cuda, cleanup_gpu):
        """Test allocating memory from pool."""
        from cudacc.memory import MemoryPool
        import numpy as np
        
        pool = MemoryPool(device_id=0)
        arr = pool.allocate(1000, dtype=np.float32)
        
        assert arr is not None
        assert arr.size == 1000
    
    @pytest.mark.cuda
    def test_free_unused(self, skip_if_no_cuda, cleanup_gpu):
        """Test freeing unused memory."""
        from cudacc.memory import MemoryPool
        
        pool = MemoryPool(device_id=0)
        pool.allocate(1000)
        
        # Should not raise
        pool.free_unused()


class TestMultiGPUMemoryManager:
    """Test suite for MultiGPUMemoryManager."""
    
    @pytest.mark.cuda
    def test_multi_gpu_manager_creation(self, skip_if_no_cuda):
        """Test creating multi-GPU manager."""
        from cudacc.memory import MultiGPUMemoryManager
        
        manager = MultiGPUMemoryManager([0])
        assert 0 in manager.pools
    
    @pytest.mark.cuda
    def test_get_pool(self, skip_if_no_cuda):
        """Test getting pool for specific device."""
        from cudacc.memory import MultiGPUMemoryManager
        
        manager = MultiGPUMemoryManager([0])
        pool = manager.get_pool(0)
        
        assert pool.device_id == 0
    
    @pytest.mark.cuda
    def test_free_all_unused(self, skip_if_no_cuda, cleanup_gpu):
        """Test freeing memory on all devices."""
        from cudacc.memory import MultiGPUMemoryManager
        
        manager = MultiGPUMemoryManager([0])
        
        # Should not raise
        manager.free_all_unused()
