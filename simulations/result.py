"""
Unified return type for Langevin simulation pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

@dataclass
class SimulationResult:
    """
    Complete result object from run_langevin_simulation_pipeline().
    
    Provides structured access to all pipeline outputs while maintaining
    backward compatibility with dict-based access (see __getitem__).
    """
    
    trajectory: np.ndarray
    """Occupancy trajectory over time (length N)."""
    
    descent_speed: float
    """Mean descent speed (negative=stable, positive=chaotic)."""
    
    violations: int = 0
    """Count of timesteps outside homeostasis tolerance."""
    
    # Optional enrichments (populated in future stages)
    attractors: List[float] = field(default_factory=list)
    """Identified attractor states (Stage 3)."""
    
    transition_times: List[float] = field(default_factory=list)
    """Times of basin transitions (Stage 3)."""
    
    dwell_times: Dict[str, float] = field(default_factory=dict)
    """Dwell time statistics by attractor (Stage 3)."""
    
    metadata: Dict = field(default_factory=dict)
    """Additional metadata for analysis."""
    
    def __getitem__(self, key: str):
        """
        Dict-like access for backward compatibility.
        
        Example:
            result["trajectory"]  # Same as result.trajectory
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"SimulationResult has no field '{key}'")
    
    def to_dict(self) -> dict:
        """Convert to dict format (for legacy code)."""
        return {
            'trajectory': self.trajectory,
            'descent_speed': self.descent_speed,
            'violations': self.violations,
            'attractors': self.attractors,
            'transition_times': self.transition_times,
            'dwell_times': self.dwell_times,
            'metadata': self.metadata,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'SimulationResult':
        """Convert from dict format (for legacy code)."""
        return SimulationResult(
            trajectory=data.get('trajectory', np.array([])),
            descent_speed=data.get('descent_speed', 0.0),
            violations=data.get('violations', 0),
            attractors=data.get('attractors', []),
            transition_times=data.get('transition_times', []),
            dwell_times=data.get('dwell_times', {}),
            metadata=data.get('metadata', {}),
        )
