"""
Domain-specific physics kernels: particle physics, fluid dynamics.

Specialized kernels for scientific computing domains.
"""

from numba import cuda
import numpy as np
import math


@cuda.jit
def particle_distance_kernel(x1, y1, z1, x2, y2, z2, distances):
    """
    Compute distances between particle pairs.
    
    Args:
        x1, y1, z1: Coordinates of first particle set.
        x2, y2, z2: Coordinates of second particle set.
        distances: Output distance array.
    """
    idx = cuda.grid(1)
    
    if idx < distances.size:
        dx = x2[idx] - x1[idx]
        dy = y2[idx] - y1[idx]
        dz = z2[idx] - z1[idx]
        
        distances[idx] = math.sqrt(dx*dx + dy*dy + dz*dz)


@cuda.jit
def invariant_mass_kernel(e1, px1, py1, pz1, e2, px2, py2, pz2, masses):
    """
    Compute invariant mass for particle pairs.
    
    Args:
        e1, px1, py1, pz1: Four-momentum of first particle.
        e2, px2, py2, pz2: Four-momentum of second particle.
        masses: Output invariant mass array.
    """
    idx = cuda.grid(1)
    
    if idx < masses.size:
        # Four-momentum addition
        e_tot = e1[idx] + e2[idx]
        px_tot = px1[idx] + px2[idx]
        py_tot = py1[idx] + py2[idx]
        pz_tot = pz1[idx] + pz2[idx]
        
        # Invariant mass: m^2 = E^2 - p^2
        p_squared = px_tot*px_tot + py_tot*py_tot + pz_tot*pz_tot
        m_squared = e_tot*e_tot - p_squared
        
        masses[idx] = math.sqrt(abs(m_squared))


@cuda.jit
def lorentz_boost_kernel(e, px, py, pz, beta_x, beta_y, beta_z, 
                         e_out, px_out, py_out, pz_out):
    """
    Apply Lorentz boost to four-momenta.
    
    Args:
        e, px, py, pz: Input four-momentum components.
        beta_x, beta_y, beta_z: Boost velocity (in units of c).
        e_out, px_out, py_out, pz_out: Output four-momentum components.
    """
    idx = cuda.grid(1)
    
    if idx < e.size:
        beta_squared = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z
        gamma = 1.0 / math.sqrt(1.0 - beta_squared)
        
        # Simplified Lorentz boost
        e_out[idx] = gamma * (e[idx] - beta_x*px[idx] - beta_y*py[idx] - beta_z*pz[idx])
        px_out[idx] = px[idx] + (gamma - 1.0) * beta_x * (beta_x*px[idx] + beta_y*py[idx] + beta_z*pz[idx]) / beta_squared - gamma * beta_x * e[idx]
        py_out[idx] = py[idx] + (gamma - 1.0) * beta_y * (beta_x*px[idx] + beta_y*py[idx] + beta_z*pz[idx]) / beta_squared - gamma * beta_y * e[idx]
        pz_out[idx] = pz[idx] + (gamma - 1.0) * beta_z * (beta_x*px[idx] + beta_y*py[idx] + beta_z*pz[idx]) / beta_squared - gamma * beta_z * e[idx]


@cuda.jit
def fluid_advection_kernel(u, v, field, dt, dx, dy, out):
    """
    Semi-Lagrangian advection for fluid simulation.
    
    Args:
        u, v: Velocity field components.
        field: Scalar field to advect.
        dt: Time step.
        dx, dy: Grid spacing.
        out: Output advected field.
    """
    i, j = cuda.grid(2)
    
    if i < field.shape[0] and j < field.shape[1]:
        # Trace back along velocity field
        x_back = i - u[i, j] * dt / dx
        y_back = j - v[i, j] * dt / dy
        
        # Clamp to grid bounds
        x_back = max(0.0, min(float(field.shape[0] - 1), x_back))
        y_back = max(0.0, min(float(field.shape[1] - 1), y_back))
        
        # Bilinear interpolation (simplified)
        i0 = int(x_back)
        j0 = int(y_back)
        
        if i0 < field.shape[0] - 1 and j0 < field.shape[1] - 1:
            fx = x_back - i0
            fy = y_back - j0
            
            out[i, j] = (
                (1 - fx) * (1 - fy) * field[i0, j0] +
                fx * (1 - fy) * field[i0 + 1, j0] +
                (1 - fx) * fy * field[i0, j0 + 1] +
                fx * fy * field[i0 + 1, j0 + 1]
            )
        else:
            out[i, j] = field[i0, j0]


def gpu_invariant_mass(e1, px1, py1, pz1, e2, px2, py2, pz2):
    """
    Compute invariant mass for particle pairs on GPU.
    
    Args:
        e1, px1, py1, pz1: Four-momentum of first particles.
        e2, px2, py2, pz2: Four-momentum of second particles.
    
    Returns:
        Array of invariant masses.
    """
    try:
        import cupy as cp
        
        # Transfer to GPU
        e1_gpu = cp.asarray(e1)
        px1_gpu = cp.asarray(px1)
        py1_gpu = cp.asarray(py1)
        pz1_gpu = cp.asarray(pz1)
        e2_gpu = cp.asarray(e2)
        px2_gpu = cp.asarray(px2)
        py2_gpu = cp.asarray(py2)
        pz2_gpu = cp.asarray(pz2)
        
        masses = cp.empty(e1_gpu.size, dtype=np.float32)
        
        threads_per_block = 256
        blocks = math.ceil(masses.size / threads_per_block)
        
        invariant_mass_kernel[blocks, threads_per_block](
            e1_gpu, px1_gpu, py1_gpu, pz1_gpu,
            e2_gpu, px2_gpu, py2_gpu, pz2_gpu,
            masses
        )
        
        return masses
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")


def gpu_particle_distance(x1, y1, z1, x2, y2, z2):
    """Compute particle-particle distances on GPU."""
    try:
        import cupy as cp
        
        x1_gpu = cp.asarray(x1)
        y1_gpu = cp.asarray(y1)
        z1_gpu = cp.asarray(z1)
        x2_gpu = cp.asarray(x2)
        y2_gpu = cp.asarray(y2)
        z2_gpu = cp.asarray(z2)
        
        distances = cp.empty(x1_gpu.size, dtype=np.float32)
        
        threads_per_block = 256
        blocks = math.ceil(distances.size / threads_per_block)
        
        particle_distance_kernel[blocks, threads_per_block](
            x1_gpu, y1_gpu, z1_gpu, x2_gpu, y2_gpu, z2_gpu, distances
        )
        
        return distances
    except ImportError:
        raise RuntimeError("CuPy is required for GPU operations")
