"""
FEniCS fluid dynamics acceleration demo.

Demonstrates GPU acceleration of computational fluid dynamics
simulations. This is a simplified example showing the concept.
"""

import numpy as np
import time


def generate_flow_field(nx=512, ny=512):
    """Generate a simple 2D flow field."""
    print(f"Generating {nx}x{ny} flow field...")
    
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    X, Y = np.meshgrid(x, y)
    
    # Simple vortex flow
    u = -np.sin(Y)
    v = np.sin(X)
    
    # Scalar field (temperature, concentration, etc.)
    field = np.sin(X) * np.cos(Y)
    
    return u.astype(np.float32), v.astype(np.float32), field.astype(np.float32)


def cpu_advection_step(u, v, field, dt=0.01):
    """
    Perform one advection step on CPU.
    
    Semi-Lagrangian advection.
    """
    nx, ny = field.shape
    
    # Create coordinate grids
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    # Trace back
    X_back = X - u * dt
    Y_back = Y - v * dt
    
    # Clamp to bounds
    X_back = np.clip(X_back, 0, nx-1)
    Y_back = np.clip(Y_back, 0, ny-1)
    
    # Simple nearest neighbor interpolation
    X_int = X_back.astype(int)
    Y_int = Y_back.astype(int)
    
    new_field = field[X_int, Y_int]
    
    return new_field


def gpu_advection_step(u, v, field, dt=0.01):
    """
    Perform one advection step on GPU.
    
    Uses CuPy for GPU acceleration.
    """
    try:
        import cupy as cp
        from cupyx.scipy import ndimage
        
        # Transfer to GPU
        u_gpu = cp.asarray(u)
        v_gpu = cp.asarray(v)
        field_gpu = cp.asarray(field)
        
        nx, ny = field_gpu.shape
        
        # Create coordinate grids
        x = cp.arange(nx)
        y = cp.arange(ny)
        X, Y = cp.meshgrid(x, y)
        
        # Trace back
        X_back = X - u_gpu * dt
        Y_back = Y - v_gpu * dt
        
        # Clamp to bounds
        X_back = cp.clip(X_back, 0, nx-1)
        Y_back = cp.clip(Y_back, 0, ny-1)
        
        # Interpolation using map_coordinates
        coords = cp.array([X_back, Y_back])
        new_field = ndimage.map_coordinates(field_gpu, coords, order=1)
        
        return new_field
        
    except ImportError as e:
        raise RuntimeError(f"GPU advection requires CuPy: {e}")


def run_simulation(advection_fn, u, v, field, steps=100):
    """Run the simulation for a number of steps."""
    current_field = field.copy()
    
    for step in range(steps):
        current_field = advection_fn(u, v, current_field)
    
    return current_field


def main():
    """Run the fluid dynamics demo."""
    print("=" * 60)
    print("cudacc Fluid Dynamics Demo (FEniCS-style)")
    print("=" * 60)
    print()
    
    # Generate flow field
    u, v, field = generate_flow_field(nx=512, ny=512)
    print(f"Flow field size: {field.shape}")
    print(f"Memory: {field.nbytes / 1024**2:.2f} MB")
    print()
    
    steps = 100
    print(f"Running {steps} advection steps...")
    print()
    
    # CPU simulation
    print("Running on CPU...")
    start = time.time()
    cpu_result = run_simulation(cpu_advection_step, u, v, field, steps=steps)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"CPU result stats: min={cpu_result.min():.4f}, max={cpu_result.max():.4f}, mean={cpu_result.mean():.4f}")
    print()
    
    # GPU simulation
    try:
        print("Running on GPU with cudacc...")
        start = time.time()
        gpu_result = run_simulation(gpu_advection_step, u, v, field, steps=steps)
        gpu_time = time.time() - start
        
        # Transfer result back to CPU for comparison
        try:
            import cupy as cp
            gpu_result_cpu = cp.asnumpy(gpu_result)
        except:
            gpu_result_cpu = gpu_result
        
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"GPU result stats: min={gpu_result_cpu.min():.4f}, max={gpu_result_cpu.max():.4f}, mean={gpu_result_cpu.mean():.4f}")
        print()
        
        # Speedup
        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.2f}x")
        print()
        
        # Verify results are similar
        diff = np.abs(cpu_result - gpu_result_cpu).mean()
        print(f"Mean absolute difference: {diff:.6f}")
        
    except Exception as e:
        print(f"GPU simulation failed: {e}")
        print("Make sure you have CUDA and CuPy installed.")
    
    print()
    print("=" * 60)
    print("Note: This is a simplified demonstration. Real FEniCS")
    print("simulations involve complex finite element methods.")
    print("=" * 60)


if __name__ == '__main__':
    main()
