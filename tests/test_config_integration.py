"""
Test that SimulationConfig is properly integrated.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.config import SimulationConfig, DEFAULT_SIMULATION_CONFIG
from simulations.colored_noise_langevin_model import run_langevin_simulation_pipeline, solve_sde

def test_config_values_used():
    """Verify that config values override defaults."""
    
    # Create custom config
    custom_config = SimulationConfig(
        total_time=10.0,  # Half the default
        dt=0.02,  # Double the default
        k_on=0.08,  # Double the default
        k_off=0.04,  # Double the default
    )
    
    # Run with custom config
    result = run_langevin_simulation_pipeline(
        omega_mut=0.5,
        haddock_score=0.1,
        config=custom_config,
        save_plot=False
    )
    
    # Trajectory should have (10.0 / 0.02) = 500 points
    expected_length = int(custom_config.total_time / custom_config.dt)
    assert len(result["trajectory"]) == expected_length, \
        f"Expected {expected_length} points, got {len(result['trajectory'])}"
    
    print(f"✅ Config integration works: trajectory length = {len(result['trajectory'])}")

def test_solve_sde_with_config():
    """Verify solve_sde accepts and uses config."""
    
    custom_config = SimulationConfig(
        total_time=5.0,
        dt=0.05,
    )
    
    # This should not raise an error
    trajectory, lambda_max = solve_sde(
        omega_mut=0.3,
        haddock_score=-10.0,
        config=custom_config
    )
    
    expected_length = int(custom_config.total_time / custom_config.dt)
    assert len(trajectory) == expected_length, \
        f"Expected {expected_length} points, got {len(trajectory)}"
    
    print(f"✅ solve_sde with config works: trajectory length = {len(trajectory)}")

def test_default_config_still_works():
    """Verify backward compatibility - calling without config uses defaults."""
    
    trajectory, lambda_max = solve_sde(
        omega_mut=0.5,
        haddock_score=-5.0
    )
    
    expected_length = int(DEFAULT_SIMULATION_CONFIG.total_time / DEFAULT_SIMULATION_CONFIG.dt)
    assert len(trajectory) == expected_length, \
        f"Expected {expected_length} points, got {len(trajectory)}"
    
    print(f"✅ Backward compatibility: default config works, trajectory length = {len(trajectory)}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST: SimulationConfig Integration")
    print("="*70)
    
    test_config_values_used()
    test_solve_sde_with_config()
    test_default_config_still_works()
    
    print("\n" + "="*70)
    print("✅ ALL CONFIG TESTS PASSED")
    print("="*70)
