"""
Test trajectory analysis module API.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.trajectory_analysis import (
    detect_attractors,
    detect_transitions,
    estimate_dwell_times,
    occupancy_statistics,
    AttractorRegion,
)

def test_detect_attractors():
    """Test attractor detection."""
    
    # Create bimodal distribution
    np.random.seed(42)
    traj = np.concatenate([
        np.random.normal(0.2, 0.05, 400),
        np.random.normal(0.8, 0.05, 400)
    ])
    traj = np.clip(traj, 0.01, 0.99)
    
    attractors = detect_attractors(traj, n_bins=10, threshold=0.02)
    
    assert len(attractors) >= 1, "Should detect at least one attractor"
    for a in attractors:
        assert isinstance(a, AttractorRegion)
        assert 0 <= a.center <= 1
        assert a.bounds[0] < a.bounds[1]
    
    print(f"✅ Attractor detection: found {len(attractors)} basins")
    for i, a in enumerate(attractors):
        print(f"   Basin {i}: center={a.center:.3f}, bounds={a.bounds}")

def test_detect_transitions():
    """Test transition detection."""
    
    # Create trajectory that transitions between two states
    traj = np.concatenate([
        np.ones(50) * 0.2,   # State 1
        np.ones(50) * 0.8,   # State 2
        np.ones(50) * 0.2,   # State 1 again
    ])
    
    attractors = [
        AttractorRegion(0.2, (0.15, 0.25)),
        AttractorRegion(0.8, (0.75, 0.85)),
    ]
    
    transitions = detect_transitions(traj, attractors, min_dwell=20)
    
    # Should detect transitions from state 1→2 and 2→1
    assert len(transitions) >= 1, "Should detect at least one transition"
    
    print(f"✅ Transition detection: found {len(transitions)} transitions")
    for t_idx, from_basin, to_basin in transitions:
        print(f"   Time {t_idx}: basin {from_basin} → {to_basin}")

def test_dwell_times():
    """Test dwell time estimation."""
    
    # Same trajectory as above
    traj = np.concatenate([
        np.ones(100) * 0.2,
        np.ones(100) * 0.8,
    ])
    
    attractors = [
        AttractorRegion(0.2, (0.15, 0.25)),
        AttractorRegion(0.8, (0.75, 0.85)),
    ]
    
    dwell_times = estimate_dwell_times(traj, attractors)
    
    assert len(dwell_times) == 2, "Should have dwell times for 2 basins"
    assert all(len(v) > 0 for v in dwell_times.values()), "All basins should have dwells"
    
    print(f"✅ Dwell time estimation:")
    for basin_idx, dwells in dwell_times.items():
        print(f"   Basin {basin_idx}: {len(dwells)} visits, mean dwell={np.mean(dwells):.1f}")

def test_occupancy_statistics():
    """Test occupancy statistics."""
    
    np.random.seed(42)
    traj = np.random.uniform(0, 1, 1000)
    
    stats = occupancy_statistics(traj)
    
    # Check expected keys
    expected_keys = ['mean', 'std', 'median', 'min', 'max', 'range', 'q25', 'q75']
    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"
    
    # Check reasonable values
    assert 0 <= stats['mean'] <= 1, "Mean should be in [0, 1]"
    assert stats['min'] <= stats['median'] <= stats['max'], "Median should be between min/max"
    
    print(f"✅ Occupancy statistics:")
    print(f"   Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"   Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"   Quantiles: Q25={stats['q25']:.3f}, Q75={stats['q75']:.3f}")

def test_occupancy_stats_with_attractors():
    """Test occupancy statistics with attractor basins."""
    
    # Bimodal distribution
    np.random.seed(42)
    traj = np.concatenate([
        np.random.normal(0.2, 0.05, 500),
        np.random.normal(0.8, 0.05, 500)
    ])
    traj = np.clip(traj, 0.01, 0.99)
    
    attractors = [
        AttractorRegion(0.2, (0.10, 0.30)),
        AttractorRegion(0.8, (0.70, 0.90)),
    ]
    
    stats = occupancy_statistics(traj, attractors)
    
    # Should have basin fractions
    assert 'basin_0_fraction' in stats
    assert 'basin_1_fraction' in stats
    
    # Fractions should sum close to 1
    total_fraction = stats['basin_0_fraction'] + stats['basin_1_fraction']
    assert 0.9 < total_fraction <= 1.0, f"Fractions should sum ~1, got {total_fraction}"
    
    print(f"✅ Statistics with attractors:")
    print(f"   Basin 0 fraction: {stats['basin_0_fraction']:.3f}")
    print(f"   Basin 1 fraction: {stats['basin_1_fraction']:.3f}")

def test_stage3_functions_raise():
    """Test that Stage 3 functions raise NotImplementedError."""
    
    from simulations.trajectory_analysis import (
        compute_lyapunov_exponent,
        bifurcation_scan
    )
    
    traj = np.random.uniform(0, 1, 1000)
    
    try:
        compute_lyapunov_exponent(traj)
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        print(f"✅ Stage 3 function protected: compute_lyapunov_exponent")
    
    try:
        bifurcation_scan(np.linspace(0, 1, 10), None)
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        print(f"✅ Stage 3 function protected: bifurcation_scan")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST: Trajectory Analysis Module API")
    print("="*70)
    
    test_detect_attractors()
    test_detect_transitions()
    test_dwell_times()
    test_occupancy_statistics()
    test_occupancy_stats_with_attractors()
    test_stage3_functions_raise()
    
    print("\n" + "="*70)
    print("✅ ALL TRAJECTORY ANALYSIS TESTS PASSED")
    print("="*70)
