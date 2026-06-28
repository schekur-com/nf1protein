"""
Test RNG reproducibility - same seed must produce identical output.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations import random as sim_random
from simulations.colored_noise_langevin_model import run_langevin_simulation_pipeline
from simulations.config import DEFAULT_SIMULATION_CONFIG

def test_rng_reproducibility():
    """Verify that same seed produces identical trajectories."""
    
    SEED = 12345
    
    # Run 1
    sim_random.set_seed(SEED)
    result1 = run_langevin_simulation_pipeline(
        omega_mut=0.5,
        haddock_score=-10.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    traj1 = result1["trajectory"]
    descent1 = result1["descent_speed"]
    
    # Run 2 with same seed
    sim_random.set_seed(SEED)
    result2 = run_langevin_simulation_pipeline(
        omega_mut=0.5,
        haddock_score=-10.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    traj2 = result2["trajectory"]
    descent2 = result2["descent_speed"]
    
    # Verify identical
    assert np.allclose(traj1, traj2, rtol=1e-15), \
        f"Trajectories differ! Max diff: {np.max(np.abs(traj1 - traj2))}"
    assert np.isclose(descent1, descent2, rtol=1e-15), \
        f"Descent metrics differ! {descent1} vs {descent2}"
    
    print(f"✅ Reproducibility verified: seed {SEED} produces identical output")
    print(f"   Trajectory length: {len(traj1)}")
    print(f"   Descent metric: {descent1:.6f}")

def test_rng_different_with_different_seed():
    """Verify that different seeds produce different trajectories."""
    
    # Run with seed 1
    sim_random.set_seed(111)
    result1 = run_langevin_simulation_pipeline(
        omega_mut=0.5,
        haddock_score=-10.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    traj1 = result1["trajectory"]
    
    # Run with different seed
    sim_random.set_seed(222)
    result2 = run_langevin_simulation_pipeline(
        omega_mut=0.5,
        haddock_score=-10.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    traj2 = result2["trajectory"]
    
    # Should be different
    assert not np.allclose(traj1, traj2, rtol=1e-5), \
        "Different seeds should produce different trajectories!"
    
    max_diff = np.max(np.abs(traj1 - traj2))
    print(f"✅ Different seeds verified: max trajectory difference = {max_diff:.6f}")

def test_default_behavior_unchanged():
    """Verify that default (no seed) still works."""
    
    sim_random.reset()  # Clear seed
    
    # Should not raise
    result = run_langevin_simulation_pipeline(
        omega_mut=0.3,
        haddock_score=-5.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    
    assert len(result["trajectory"]) > 0
    print(f"✅ Default behavior (no seed): works correctly")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST: RNG Centralization & Reproducibility")
    print("="*70)
    
    test_rng_reproducibility()
    test_rng_different_with_different_seed()
    test_default_behavior_unchanged()
    
    print("\n" + "="*70)
    print("✅ ALL RNG TESTS PASSED")
    print("="*70)
