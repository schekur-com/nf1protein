"""
Trajectory analysis module for Stage 3 scientific work.

This module provides tools for analyzing occupancy trajectories:
- Attractor detection (basin identification)
- Transition detection (basin hopping)
- Dwell time estimation
- Occupancy statistics

Currently NOT used by mutation_robustness_screen. Created as API
for Stage 3 bifurcation analysis and Lyapunov computation.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class AttractorRegion:
    """Definition of an attractor basin."""
    center: float
    """Central occupancy value."""
    
    bounds: Tuple[float, float]
    """(lower, upper) bounds of basin."""
    
    residence_time: float = 0.0
    """Total time spent in this basin."""
    
    visit_count: int = 0
    """Number of entries to this basin."""


def detect_attractors(
    trajectory: np.ndarray,
    method: str = "histogram",
    n_bins: int = 10,
    threshold: float = 0.05
) -> List[AttractorRegion]:
    """
    Detect attractor basins from trajectory.
    
    Uses histogram-based or kernel density estimation to identify
    regions where the system spends significant time.
    
    Args:
        trajectory: 1D array of occupancy values
        method: "histogram" or "kde" (kernel density estimation)
        n_bins: Number of bins for histogram
        threshold: Minimum fraction of time to define attractor
        
    Returns:
        List of AttractorRegion objects sorted by center
        
    Examples:
        >>> traj = np.random.uniform(0, 1, 1000)
        >>> attractors = detect_attractors(traj)
        >>> print(f"Found {len(attractors)} basins")
    """
    if method == "histogram":
        counts, bin_edges = np.histogram(trajectory, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize to get fraction
        fractions = counts / len(trajectory)
        
        # Find bins above threshold
        significant = np.where(fractions >= threshold)[0]
        
        attractors = []
        for idx in significant:
            attractors.append(AttractorRegion(
                center=float(bin_centers[idx]),
                bounds=(float(bin_edges[idx]), float(bin_edges[idx + 1]))
            ))
        
        # Sort by center
        attractors.sort(key=lambda x: x.center)
        return attractors
    
    elif method == "kde":
        # Placeholder for KDE-based detection (Stage 3)
        raise NotImplementedError("KDE method for Stage 3")
    
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_transitions(
    trajectory: np.ndarray,
    attractors: List[AttractorRegion],
    min_dwell: int = 10
) -> List[Tuple[int, int, int]]:
    """
    Detect transitions between attractor basins.
    
    A transition is detected when the system moves from one basin
    to another and stays there for at least min_dwell timesteps.
    
    Args:
        trajectory: 1D array of occupancy values
        attractors: List of AttractorRegion objects
        min_dwell: Minimum timesteps required in basin to count as valid
        
    Returns:
        List of (time_index, from_basin, to_basin) tuples
        
    Examples:
        >>> transitions = detect_transitions(traj, attractors)
        >>> print(f"Found {len(transitions)} transitions")
    """
    if len(attractors) == 0:
        return []
    
    # Assign each trajectory point to nearest attractor
    basin_sequence = np.zeros(len(trajectory), dtype=int)
    for i, val in enumerate(trajectory):
        distances = [abs(val - a.center) for a in attractors]
        basin_sequence[i] = np.argmin(distances)
    
    # Find transitions (change in basin)
    transitions = []
    for i in range(len(basin_sequence) - 1):
        if basin_sequence[i] != basin_sequence[i + 1]:
            # Check if new basin lasts at least min_dwell steps
            new_basin = basin_sequence[i + 1]
            dwell_count = 1
            for j in range(i + 2, len(basin_sequence)):
                if basin_sequence[j] == new_basin:
                    dwell_count += 1
                else:
                    break
            
            if dwell_count >= min_dwell:
                transitions.append((i, basin_sequence[i], new_basin))
    
    return transitions


def estimate_dwell_times(
    trajectory: np.ndarray,
    attractors: List[AttractorRegion]
) -> Dict[int, List[float]]:
    """
    Estimate dwell times in each basin.
    
    For each attractor, compute list of consecutive times spent in that basin.
    
    Args:
        trajectory: 1D array of occupancy values
        attractors: List of AttractorRegion objects
        
    Returns:
        Dict mapping basin index to list of dwell durations
        
    Examples:
        >>> dwell_times = estimate_dwell_times(traj, attractors)
        >>> for basin_idx, dwells in dwell_times.items():
        ...     print(f"Basin {basin_idx}: mean dwell = {np.mean(dwells):.2f}")
    """
    if len(attractors) == 0:
        return {}
    
    # Assign each point to basin
    basin_sequence = np.zeros(len(trajectory), dtype=int)
    for i, val in enumerate(trajectory):
        distances = [abs(val - a.center) for a in attractors]
        basin_sequence[i] = np.argmin(distances)
    
    # Compute dwell times
    dwell_times = {i: [] for i in range(len(attractors))}
    current_basin = basin_sequence[0]
    current_dwell = 1
    
    for i in range(1, len(basin_sequence)):
        if basin_sequence[i] == current_basin:
            current_dwell += 1
        else:
            # Basin change
            dwell_times[current_basin].append(current_dwell)
            current_basin = basin_sequence[i]
            current_dwell = 1
    
    # Don't forget last dwell
    dwell_times[current_basin].append(current_dwell)
    
    return dwell_times


def occupancy_statistics(
    trajectory: np.ndarray,
    attractors: List[AttractorRegion] = None
) -> Dict[str, float]:
    """
    Compute occupancy statistics from trajectory.
    
    Args:
        trajectory: 1D array of occupancy values
        attractors: Optional list of attractors for basin statistics
        
    Returns:
        Dict with statistics: mean, std, min, max, skewness, kurtosis, etc.
        
    Examples:
        >>> stats = occupancy_statistics(traj)
        >>> print(f"Mean occupancy: {stats['mean']:.3f}")
    """
    trajectory = np.asarray(trajectory)
    
    result = {
        'mean': float(np.mean(trajectory)),
        'std': float(np.std(trajectory)),
        'median': float(np.median(trajectory)),
        'min': float(np.min(trajectory)),
        'max': float(np.max(trajectory)),
        'range': float(np.max(trajectory) - np.min(trajectory)),
    }
    
    # Compute quantiles
    for q in [0.25, 0.75]:
        result[f'q{int(100*q)}'] = float(np.quantile(trajectory, q))
    
    # If attractors provided, add basin statistics
    if attractors is not None and len(attractors) > 0:
        for idx, attractor in enumerate(attractors):
            in_basin = (trajectory >= attractor.bounds[0]) & \
                       (trajectory < attractor.bounds[1])
            result[f'basin_{idx}_fraction'] = float(np.mean(in_basin))
    
    return result


# Placeholder for future Stage 3 analysis
def compute_lyapunov_exponent(trajectory: np.ndarray, method: str = "wolf") -> float:
    """
    Compute largest Lyapunov exponent from trajectory.
    
    Stage 3 only. Not used in Stage 2.
    
    Args:
        trajectory: 1D array
        method: "wolf" (Benettin-Wolf), "qr", or "rosenstein"
        
    Returns:
        Estimated maximum Lyapunov exponent
        
    Raises:
        NotImplementedError: Feature is Stage 3
    """
    raise NotImplementedError("Lyapunov computation is Stage 3 work")


def bifurcation_scan(
    parameter_values: np.ndarray,
    trajectory_generator,
    n_transient: int = 1000
) -> Dict:
    """
    Scan bifurcation diagram by varying a parameter.
    
    Stage 3 only. Not used in Stage 2.
    
    Args:
        parameter_values: Array of parameter values to scan
        trajectory_generator: Function to generate trajectory for each parameter
        n_transient: Number of initial steps to discard (transient)
        
    Returns:
        Bifurcation data (attractors vs parameter)
        
    Raises:
        NotImplementedError: Feature is Stage 3
    """
    raise NotImplementedError("Bifurcation analysis is Stage 3 work")
