"""
Device detection and multi-GPU topology management.

Provides utilities for detecting CUDA devices, querying their properties,
and managing multi-GPU topologies.
"""

from typing import List, Dict, Optional, Tuple
import logging


logger = logging.getLogger(__name__)


def detect_devices() -> List[int]:
    """
    Detect all available CUDA devices.
    
    Returns:
        List of device IDs (e.g., [0, 1, 2] for 3 GPUs).
    """
    try:
        import cupy as cp
        num_devices = cp.cuda.runtime.getDeviceCount()
        return list(range(num_devices))
    except Exception as e:
        logger.warning(f"CUDA device detection failed: {e}")
        return []


def get_device_properties(device_id: int) -> Dict[str, any]:
    """
    Get properties of a specific CUDA device.
    
    Args:
        device_id: CUDA device ID.
    
    Returns:
        Dictionary with device properties.
    """
    try:
        import cupy as cp
        
        with cp.cuda.Device(device_id):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            
            return {
                'name': props['name'].decode('utf-8'),
                'compute_capability': (props['major'], props['minor']),
                'total_memory': props['totalGlobalMem'],
                'multiprocessor_count': props['multiProcessorCount'],
                'max_threads_per_block': props['maxThreadsPerBlock'],
                'max_threads_per_multiprocessor': props['maxThreadsPerMultiProcessor'],
                'warp_size': props['warpSize'],
                'device_id': device_id,
            }
    except Exception as e:
        logger.error(f"Failed to get properties for device {device_id}: {e}")
        return {}


def get_all_device_properties() -> List[Dict[str, any]]:
    """
    Get properties for all available devices.
    
    Returns:
        List of device property dictionaries.
    """
    devices = detect_devices()
    return [get_device_properties(dev_id) for dev_id in devices]


def validate_devices(requested: List[int], available: List[int]) -> bool:
    """
    Validate that requested devices are available.
    
    Args:
        requested: List of requested device IDs.
        available: List of available device IDs.
    
    Returns:
        True if all requested devices are available.
    
    Raises:
        ValueError: If any requested device is not available.
    """
    for device_id in requested:
        if device_id not in available:
            raise ValueError(
                f"Device {device_id} not available. "
                f"Available devices: {available}"
            )
    return True


def get_peer_access_matrix() -> Optional[List[List[bool]]]:
    """
    Get peer-to-peer access matrix for multi-GPU systems.
    
    Returns:
        Matrix where element [i][j] is True if device i can access device j.
        Returns None if only one GPU or if checking fails.
    """
    try:
        import cupy as cp
        
        devices = detect_devices()
        if len(devices) <= 1:
            return None
        
        n = len(devices)
        matrix = [[False] * n for _ in range(n)]
        
        for i in devices:
            matrix[i][i] = True  # Device can access itself
            for j in devices:
                if i != j:
                    try:
                        can_access = cp.cuda.runtime.deviceCanAccessPeer(i, j)
                        matrix[i][j] = bool(can_access)
                    except Exception:
                        matrix[i][j] = False
        
        return matrix
    except Exception as e:
        logger.warning(f"Failed to get peer access matrix: {e}")
        return None


def print_device_info():
    """Print information about all available CUDA devices."""
    devices = detect_devices()
    
    if not devices:
        print("No CUDA devices found.")
        return
    
    print(f"Found {len(devices)} CUDA device(s):")
    print()
    
    for device_id in devices:
        props = get_device_properties(device_id)
        if props:
            print(f"Device {device_id}: {props['name']}")
            print(f"  Compute Capability: {props['compute_capability'][0]}.{props['compute_capability'][1]}")
            print(f"  Total Memory: {props['total_memory'] / 1024**3:.2f} GB")
            print(f"  Multiprocessors: {props['multiprocessor_count']}")
            print(f"  Max Threads/Block: {props['max_threads_per_block']}")
            print()
    
    # Print peer access matrix for multi-GPU systems
    if len(devices) > 1:
        matrix = get_peer_access_matrix()
        if matrix:
            print("Peer-to-Peer Access Matrix:")
            print("  ", "  ".join([f"GPU{i}" for i in devices]))
            for i, row in enumerate(matrix):
                status = "  ".join(["Yes " if val else "No  " for val in row])
                print(f"GPU{i}: {status}")
            print()


def select_best_device() -> int:
    """
    Select the best available device based on free memory.
    
    Returns:
        Device ID of the best device.
    
    Raises:
        RuntimeError: If no devices are available.
    """
    try:
        import cupy as cp
        
        devices = detect_devices()
        if not devices:
            raise RuntimeError("No CUDA devices available")
        
        best_device = 0
        max_free_memory = 0
        
        for device_id in devices:
            with cp.cuda.Device(device_id):
                free_mem, _ = cp.cuda.runtime.memGetInfo()
                if free_mem > max_free_memory:
                    max_free_memory = free_mem
                    best_device = device_id
        
        return best_device
    except Exception as e:
        logger.error(f"Failed to select best device: {e}")
        return 0


def set_device(device_id: int):
    """
    Set the active CUDA device.
    
    Args:
        device_id: Device ID to activate.
    """
    try:
        import cupy as cp
        cp.cuda.Device(device_id).use()
        logger.info(f"Set active device to GPU {device_id}")
    except Exception as e:
        logger.error(f"Failed to set device {device_id}: {e}")
        raise
