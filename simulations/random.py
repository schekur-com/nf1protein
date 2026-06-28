"""
Centralized random number generation for simulations.

All random draws should go through this module to ensure:
1. Reproducibility (single seed source)
2. Auditability (can log all RNG calls if needed)
3. Testability (can inject fixed seeds for tests)
"""

import numpy as np
from typing import Optional

# Global RNG instance
_GLOBAL_RNG: Optional[np.random.Generator] = None
_SEED: Optional[int] = None

def set_seed(seed: int) -> None:
    """
    Set the global RNG seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    global _GLOBAL_RNG, _SEED
    _SEED = seed
    _GLOBAL_RNG = np.random.default_rng(seed)

def get_rng() -> np.random.Generator:
    """
    Get the global RNG instance.
    
    If no seed has been set, initializes with default behavior (random).
    
    Returns:
        np.random.Generator instance
    """
    global _GLOBAL_RNG
    if _GLOBAL_RNG is None:
        _GLOBAL_RNG = np.random.default_rng()
    return _GLOBAL_RNG

def get_seed() -> Optional[int]:
    """
    Get the current seed if one was explicitly set.
    
    Returns:
        int or None
    """
    return _SEED

# Convenience functions
def normal(loc=0.0, scale=1.0, size=None):
    """Sample from normal distribution."""
    return get_rng().normal(loc, scale, size)

def uniform(low=0.0, high=1.0, size=None):
    """Sample from uniform distribution."""
    return get_rng().uniform(low, high, size)

def random(size=None):
    """Sample uniform [0, 1)."""
    return get_rng().random(size)

def exponential(scale=1.0, size=None):
    """Sample from exponential distribution."""
    return get_rng().exponential(scale, size)

def binomial(n, p, size=None):
    """Sample from binomial distribution."""
    return get_rng().binomial(n, p, size)

def reset():
    """Reset RNG to uninitialized state (for testing)."""
    global _GLOBAL_RNG, _SEED
    _GLOBAL_RNG = None
    _SEED = None
