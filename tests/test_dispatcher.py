"""
Tests for the kernel dispatcher.
"""

import pytest


class TestKernelDispatcher:
    """Test suite for the kernel dispatcher."""
    
    def test_dispatcher_creation(self):
        """Test creating a dispatcher."""
        from cudacc.dispatcher import KernelDispatcher
        
        dispatcher = KernelDispatcher()
        assert dispatcher is not None
    
    def test_register_kernel(self):
        """Test registering a kernel."""
        from cudacc.dispatcher import KernelDispatcher, OperationType
        
        dispatcher = KernelDispatcher()
        
        # Mock kernel and pattern
        def mock_kernel(arr):
            return arr * 2
        
        def mock_pattern(args):
            return True
        
        dispatcher.register(
            "test_op",
            mock_pattern,
            mock_kernel,
            OperationType.TRANSFORM
        )
        
        # Check dispatch works
        kernel = dispatcher.dispatch("test_op", [1, 2, 3])
        assert kernel == mock_kernel
    
    def test_dispatch_not_found(self):
        """Test dispatching non-existent operation."""
        from cudacc.dispatcher import KernelDispatcher
        
        dispatcher = KernelDispatcher()
        
        kernel = dispatcher.dispatch("nonexistent_op")
        assert kernel is None
    
    def test_pattern_matching(self):
        """Test pattern matching for dispatch."""
        from cudacc.dispatcher import KernelDispatcher, OperationType
        
        dispatcher = KernelDispatcher()
        
        def kernel_small(arr):
            return "small"
        
        def kernel_large(arr):
            return "large"
        
        def pattern_small(args):
            arr = args[0][0]
            return len(arr) < 100
        
        def pattern_large(args):
            arr = args[0][0]
            return len(arr) >= 100
        
        dispatcher.register("sum", pattern_small, kernel_small, OperationType.REDUCTION)
        dispatcher.register("sum", pattern_large, kernel_large, OperationType.REDUCTION)
        
        # Test small array
        kernel = dispatcher.dispatch("sum", [1, 2, 3])
        assert kernel == kernel_small
        
        # Test large array
        kernel = dispatcher.dispatch("sum", list(range(200)))
        assert kernel == kernel_large
    
    def test_global_dispatcher(self):
        """Test getting global dispatcher."""
        from cudacc.dispatcher import get_dispatcher
        
        dispatcher1 = get_dispatcher()
        dispatcher2 = get_dispatcher()
        
        # Should be the same instance
        assert dispatcher1 is dispatcher2
