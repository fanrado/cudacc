"""
Registry of supported packages and their acceleration bridges.

Maintains the mapping between Python packages (NumPy, SciPy, etc.)
and their corresponding GPU acceleration bridges.
"""

from typing import Any, Dict, Optional, Type
from abc import ABC, abstractmethod


class PackageBridge(ABC):
    """
    Abstract base class for package-specific acceleration bridges.
    
    Each supported package has a bridge that knows how to intercept
    and accelerate its operations.
    """
    
    @abstractmethod
    def apply(self, pkg: Any, devices: list[int]) -> Any:
        """
        Apply GPU acceleration to the package.
        
        Args:
            pkg: The package to accelerate.
            devices: List of GPU device IDs to use.
        
        Returns:
            Accelerated version of the package.
        """
        pass
    
    @abstractmethod
    def supports(self, pkg: Any) -> bool:
        """
        Check if this bridge supports the given package.
        
        Args:
            pkg: The package to check.
        
        Returns:
            True if this bridge can accelerate the package.
        """
        pass


class PackageRegistry:
    """
    Registry mapping packages to their acceleration bridges.
    """
    
    def __init__(self):
        self._bridges: Dict[str, Type[PackageBridge]] = {}
    
    def register(self, package_name: str, bridge_class: Type[PackageBridge]):
        """
        Register a bridge for a package.
        
        Args:
            package_name: Name of the package (e.g., "numpy").
            bridge_class: The bridge class for this package.
        """
        self._bridges[package_name] = bridge_class
    
    def get_bridge(self, pkg: Any) -> Optional[PackageBridge]:
        """
        Get the appropriate bridge for a package.
        
        Args:
            pkg: The package instance.
        
        Returns:
            An instance of the appropriate bridge, or None if not supported.
        """
        pkg_name = getattr(pkg, '__name__', None)
        if pkg_name is None:
            return None
        
        # Try exact match first
        if pkg_name in self._bridges:
            bridge_class = self._bridges[pkg_name]
            bridge = bridge_class()
            if bridge.supports(pkg):
                return bridge
        
        # Try checking all bridges
        for bridge_class in self._bridges.values():
            bridge = bridge_class()
            if bridge.supports(pkg):
                return bridge
        
        return None
    
    def list_supported_packages(self) -> list[str]:
        """Get list of all supported package names."""
        return list(self._bridges.keys())


# Global registry instance
_global_registry = PackageRegistry()


def register_bridge(package_name: str, bridge_class: Type[PackageBridge]):
    """Register a bridge in the global registry."""
    _global_registry.register(package_name, bridge_class)


def get_package_bridge(pkg: Any) -> Optional[PackageBridge]:
    """Get the bridge for a package from the global registry."""
    return _global_registry.get_bridge(pkg)


def list_supported_packages() -> list[str]:
    """List all supported packages."""
    return _global_registry.list_supported_packages()
