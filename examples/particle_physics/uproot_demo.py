"""
Uproot/Awkward acceleration demo for HEP analysis.

Demonstrates GPU acceleration of High Energy Physics data analysis
using uproot and awkward arrays.
"""

import numpy as np
import time


def generate_mock_events(n_events=100000):
    """
    Generate mock particle physics events.
    
    Each event contains particle four-momenta (E, px, py, pz).
    """
    print(f"Generating {n_events:,} mock events...")
    
    # Random four-momenta for particles
    events = {
        'E': np.random.uniform(1, 100, n_events).astype(np.float32),
        'px': np.random.randn(n_events).astype(np.float32) * 10,
        'py': np.random.randn(n_events).astype(np.float32) * 10,
        'pz': np.random.randn(n_events).astype(np.float32) * 20,
    }
    
    # Add transverse momentum
    events['pt'] = np.sqrt(events['px']**2 + events['py']**2)
    
    return events


def cpu_analysis(events):
    """Perform analysis on CPU."""
    # Filter by pT
    mask = (events['pt'] > 5.0) & (events['pt'] < 50.0)
    filtered_count = np.sum(mask)
    
    # Compute invariant mass for pairs
    # Simplified: just take first half with second half
    n = len(events['E']) // 2
    
    e1 = events['E'][:n]
    px1 = events['px'][:n]
    py1 = events['py'][:n]
    pz1 = events['pz'][:n]
    
    e2 = events['E'][n:2*n]
    px2 = events['px'][n:2*n]
    py2 = events['py'][n:2*n]
    pz2 = events['pz'][n:2*n]
    
    # Invariant mass calculation
    e_tot = e1 + e2
    px_tot = px1 + px2
    py_tot = py1 + py2
    pz_tot = pz1 + pz2
    
    p_squared = px_tot**2 + py_tot**2 + pz_tot**2
    m_squared = e_tot**2 - p_squared
    masses = np.sqrt(np.abs(m_squared))
    
    # Histogram
    hist, _ = np.histogram(masses, bins=50, range=(0, 200))
    
    return {
        'filtered_count': filtered_count,
        'mean_mass': np.mean(masses),
        'histogram_peak': np.argmax(hist),
    }


def gpu_analysis(events):
    """Perform analysis on GPU using cudacc."""
    from cudacc import accelerate
    
    try:
        # Try to accelerate uproot/awkward (may not be installed)
        import cupy as cp
        
        # Transfer data to GPU
        events_gpu = {
            key: cp.asarray(val) for key, val in events.items()
        }
        
        # Filter by pT
        mask = (events_gpu['pt'] > 5.0) & (events_gpu['pt'] < 50.0)
        filtered_count = int(cp.sum(mask))
        
        # Compute invariant mass
        from cudacc.kernels.physics import gpu_invariant_mass
        
        n = len(events['E']) // 2
        
        e1 = events_gpu['E'][:n]
        px1 = events_gpu['px'][:n]
        py1 = events_gpu['py'][:n]
        pz1 = events_gpu['pz'][:n]
        
        e2 = events_gpu['E'][n:2*n]
        px2 = events_gpu['px'][n:2*n]
        py2 = events_gpu['py'][n:2*n]
        pz2 = events_gpu['pz'][n:2*n]
        
        masses = gpu_invariant_mass(e1, px1, py1, pz1, e2, px2, py2, pz2)
        
        # Histogram (using CuPy)
        hist, _ = cp.histogram(masses, bins=50, range=(0, 200))
        
        return {
            'filtered_count': filtered_count,
            'mean_mass': float(cp.mean(masses)),
            'histogram_peak': int(cp.argmax(hist)),
        }
        
    except ImportError as e:
        raise RuntimeError(f"GPU analysis requires CuPy: {e}")


def main():
    """Run the HEP analysis demo."""
    print("=" * 60)
    print("cudacc HEP Analysis Demo (uproot/awkward)")
    print("=" * 60)
    print()
    
    # Generate mock data
    events = generate_mock_events(n_events=1_000_000)
    print(f"Generated {len(events['E']):,} events")
    print()
    
    # CPU analysis
    print("Running CPU analysis...")
    start = time.time()
    cpu_result = cpu_analysis(events)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"CPU result: {cpu_result}")
    print()
    
    # GPU analysis
    try:
        print("Running GPU analysis with cudacc...")
        start = time.time()
        gpu_result = gpu_analysis(events)
        gpu_time = time.time() - start
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"GPU result: {gpu_result}")
        print()
        
        # Speedup
        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"GPU analysis failed: {e}")
        print("Make sure you have CUDA and CuPy installed.")
    
    print()
    print("=" * 60)
    print("Note: This demo uses mock data. For real HEP analysis,")
    print("use uproot to read ROOT files and awkward for jagged arrays.")
    print("=" * 60)


if __name__ == '__main__':
    main()
