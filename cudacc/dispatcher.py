"""
Pattern matching and kernel dispatch table.

Maps CPU operations to their GPU kernel implementations based on
operation type, array shapes, and data types.
"""

from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OperationType(Enum):
    """Categories of operations that can be dispatched."""
    REDUCTION = "reduction"
    TRANSFORM = "transform"
    LINALG = "linalg"
    PHYSICS = "physics"


@dataclass
class DispatchRule:
    """A single dispatch rule mapping conditions to a kernel."""
    operation: str
    pattern: Callable[[Any], bool]
    kernel: Callable
    operation_type: OperationType


class KernelDispatcher:
    """
    Manages the dispatch table for GPU kernel selection.
    
    Attributes:
        _dispatch_table: Dictionary mapping operation names to dispatch rules.
    """
    
    def __init__(self):
        self._dispatch_table: Dict[str, list[DispatchRule]] = {}
    
    def register(
        self,
        operation: str,
        pattern: Callable[[Any], bool],
        kernel: Callable,
        operation_type: OperationType
    ):
        """
        Register a new dispatch rule.
        
        Args:
            operation: Name of the operation (e.g., "sum", "matmul").
            pattern: Function that returns True if this kernel should be used.
            kernel: The GPU kernel function to dispatch to.
            operation_type: Category of operation.
        """
        rule = DispatchRule(operation, pattern, kernel, operation_type)
        
        if operation not in self._dispatch_table:
            self._dispatch_table[operation] = []
        
        self._dispatch_table[operation].append(rule)
    
    def dispatch(self, operation: str, *args, **kwargs) -> Optional[Callable]:
        """
        Find the appropriate kernel for the given operation and arguments.
        
        Args:
            operation: Name of the operation.
            *args: Arguments to the operation.
            **kwargs: Keyword arguments to the operation.
        
        Returns:
            The matched kernel function, or None if no match found.
        """
        if operation not in self._dispatch_table:
            return None
        
        # Try each rule until we find a match
        for rule in self._dispatch_table[operation]:
            try:
                if rule.pattern((args, kwargs)):
                    return rule.kernel
            except Exception:
                # Pattern matching failed, try next rule
                continue
        
        return None


# Global dispatcher instance
_global_dispatcher = KernelDispatcher()


def get_dispatcher() -> KernelDispatcher:
    """Get the global kernel dispatcher instance."""
    return _global_dispatcher
