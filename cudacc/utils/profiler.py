"""
Optional timing and memory profiling hooks.

Provides utilities for profiling GPU operations including:
- Execution time measurement
- Memory usage tracking
- Kernel launch statistics
"""

from typing import Optional, Dict, List, Callable
import time
import functools
import logging


logger = logging.getLogger(__name__)


class GPUProfiler:
    """
    Context manager and decorator for profiling GPU operations.
    """
    
    def __init__(self, name: str = "operation", enabled: bool = True):
        """
        Initialize the profiler.
        
        Args:
            name: Name of the operation being profiled.
            enabled: Whether profiling is enabled.
        """
        self.name = name
        self.enabled = enabled
        self.start_time = None
        self.end_time = None
        self.memory_before = None
        self.memory_after = None
        self._results = {}
    
    def __enter__(self):
        """Start profiling."""
        if not self.enabled:
            return self
        
        try:
            import cupy as cp
            
            # Synchronize to ensure accurate timing
            cp.cuda.Stream.null.synchronize()
            
            # Record start time
            self.start_time = time.perf_counter()
            
            # Record memory usage
            self.memory_before = cp.cuda.runtime.memGetInfo()
            
        except Exception as e:
            logger.warning(f"Profiling setup failed: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and record results."""
        if not self.enabled:
            return
        
        try:
            import cupy as cp
            
            # Synchronize to ensure all operations completed
            cp.cuda.Stream.null.synchronize()
            
            # Record end time
            self.end_time = time.perf_counter()
            
            # Record memory usage
            self.memory_after = cp.cuda.runtime.memGetInfo()
            
            # Calculate metrics
            elapsed_time = self.end_time - self.start_time
            memory_used = self.memory_before[0] - self.memory_after[0]
            
            self._results = {
                'name': self.name,
                'time_seconds': elapsed_time,
                'time_ms': elapsed_time * 1000,
                'memory_used_bytes': memory_used,
                'memory_used_mb': memory_used / 1024**2,
            }
            
            # Log results
            logger.info(
                f"Profile [{self.name}]: "
                f"Time={elapsed_time*1000:.2f}ms, "
                f"Memory={memory_used/1024**2:.2f}MB"
            )
            
        except Exception as e:
            logger.warning(f"Profiling teardown failed: {e}")
    
    @property
    def results(self) -> Dict:
        """Get profiling results."""
        return self._results


def profile(name: Optional[str] = None, enabled: bool = True):
    """
    Decorator for profiling functions.
    
    Args:
        name: Name for the profiled operation (uses function name if None).
        enabled: Whether profiling is enabled.
    
    Example:
        @profile("my_gpu_function")
        def compute_on_gpu(data):
            return data ** 2
    """
    def decorator(func: Callable) -> Callable:
        operation_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with GPUProfiler(operation_name, enabled):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class MemoryTracker:
    """
    Track GPU memory usage across operations.
    """
    
    def __init__(self):
        self.snapshots: List[Dict] = []
    
    def snapshot(self, label: str = ""):
        """
        Take a memory usage snapshot.
        
        Args:
            label: Label for this snapshot.
        """
        try:
            import cupy as cp
            
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            used_mem = total_mem - free_mem
            
            snapshot = {
                'label': label,
                'timestamp': time.time(),
                'free_bytes': free_mem,
                'used_bytes': used_mem,
                'total_bytes': total_mem,
                'free_mb': free_mem / 1024**2,
                'used_mb': used_mem / 1024**2,
                'total_mb': total_mem / 1024**2,
            }
            
            self.snapshots.append(snapshot)
            
            logger.debug(
                f"Memory snapshot [{label}]: "
                f"Used={used_mem/1024**2:.2f}MB, "
                f"Free={free_mem/1024**2:.2f}MB"
            )
            
        except Exception as e:
            logger.warning(f"Memory snapshot failed: {e}")
    
    def get_peak_usage(self) -> float:
        """
        Get peak memory usage in MB.
        
        Returns:
            Peak memory usage in megabytes.
        """
        if not self.snapshots:
            return 0.0
        
        return max(s['used_mb'] for s in self.snapshots)
    
    def print_summary(self):
        """Print a summary of memory usage."""
        if not self.snapshots:
            print("No memory snapshots recorded.")
            return
        
        print("\nMemory Usage Summary:")
        print("-" * 60)
        print(f"{'Label':<30} {'Used (MB)':<15} {'Free (MB)':<15}")
        print("-" * 60)
        
        for snapshot in self.snapshots:
            print(
                f"{snapshot['label']:<30} "
                f"{snapshot['used_mb']:<15.2f} "
                f"{snapshot['free_mb']:<15.2f}"
            )
        
        print("-" * 60)
        print(f"Peak Usage: {self.get_peak_usage():.2f} MB")
        print()


def benchmark(func: Callable, *args, iterations: int = 10, warmup: int = 2, **kwargs):
    """
    Benchmark a GPU function.
    
    Args:
        func: Function to benchmark.
        *args: Arguments to pass to the function.
        iterations: Number of iterations to run.
        warmup: Number of warmup iterations (not counted).
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        Dictionary with benchmark results.
    """
    try:
        import cupy as cp
        
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
            cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            cp.cuda.Stream.null.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        import statistics
        
        return {
            'mean_time_ms': statistics.mean(times) * 1000,
            'median_time_ms': statistics.median(times) * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'std_time_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
            'iterations': iterations,
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {}
