"""
Test SimulationResult dataclass functionality.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.result import SimulationResult
from simulations.colored_noise_langevin_model import run_langevin_simulation_pipeline
from simulations.config import DEFAULT_SIMULATION_CONFIG

def test_result_is_dataclass():
    """Verify result has expected structure."""
    
    result = run_langevin_simulation_pipeline(
        omega_mut=0.5,
        haddock_score=-10.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    
    assert isinstance(result, SimulationResult), "Result should be SimulationResult"
    assert hasattr(result, 'trajectory'), "Missing trajectory"
    assert hasattr(result, 'descent_speed'), "Missing descent_speed"
    assert hasattr(result, 'violations'), "Missing violations"
    assert hasattr(result, 'metadata'), "Missing metadata"
    
    print(f"✅ SimulationResult has expected attributes")

def test_dict_backward_compatibility():
    """Verify dict-like access for legacy code."""
    
    result = run_langevin_simulation_pipeline(
        omega_mut=0.5,
        haddock_score=-10.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    
    # Old code that uses dict access should still work
    traj = result["trajectory"]
    descent = result["descent_speed"]
    viol = result["violations"]
    
    assert isinstance(traj, np.ndarray), "Dict access for trajectory broken"
    assert isinstance(descent, float), "Dict access for descent_speed broken"
    assert isinstance(viol, (int, np.integer)), "Dict access for violations broken"
    
    print(f"✅ Dict-like access backward compatible")

def test_to_dict_conversion():
    """Verify to_dict() for serialization."""
    
    result = run_langevin_simulation_pipeline(
        omega_mut=0.3,
        haddock_score=-5.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict), "to_dict() should return dict"
    assert 'trajectory' in result_dict, "Missing trajectory in dict"
    assert 'descent_speed' in result_dict, "Missing descent_speed in dict"
    assert 'metadata' in result_dict, "Missing metadata in dict"
    
    print(f"✅ to_dict() conversion works")

def test_from_dict_conversion():
    """Verify from_dict() for deserialization."""
    
    data = {
        'trajectory': np.array([0.1, 0.2, 0.3]),
        'descent_speed': -0.05,
        'violations': 10,
        'metadata': {'test': True}
    }
    
    result = SimulationResult.from_dict(data)
    
    assert isinstance(result, SimulationResult)
    assert np.allclose(result.trajectory, data['trajectory'])
    assert result.descent_speed == data['descent_speed']
    assert result.violations == data['violations']
    assert result.metadata['test'] is True
    
    print(f"✅ from_dict() conversion works")

def test_metadata_enrichment():
    """Verify metadata is preserved."""
    
    result = run_langevin_simulation_pipeline(
        omega_mut=0.7,
        haddock_score=-20.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    
    assert 'omega_mut' in result.metadata, "omega_mut missing from metadata"
    assert 'haddock_score' in result.metadata, "haddock_score missing from metadata"
    assert 'theta_native' in result.metadata, "theta_native missing from metadata"
    
    assert result.metadata['omega_mut'] == 0.7
    assert result.metadata['haddock_score'] == -20.0
    
    print(f"✅ Metadata enrichment works")

def test_stage3_fields_available():
    """Verify Stage 3 fields are pre-allocated."""
    
    result = run_langevin_simulation_pipeline(
        omega_mut=0.5,
        haddock_score=-10.0,
        config=DEFAULT_SIMULATION_CONFIG,
        save_plot=False
    )
    
    # These should be empty but accessible (for Stage 3)
    assert hasattr(result, 'attractors'), "attractors field missing"
    assert hasattr(result, 'transition_times'), "transition_times field missing"
    assert hasattr(result, 'dwell_times'), "dwell_times field missing"
    
    assert isinstance(result.attractors, list), "attractors should be list"
    assert isinstance(result.transition_times, list), "transition_times should be list"
    assert isinstance(result.dwell_times, dict), "dwell_times should be dict"
    
    print(f"✅ Stage 3 fields pre-allocated")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST: SimulationResult Dataclass")
    print("="*70)
    
    test_result_is_dataclass()
    test_dict_backward_compatibility()
    test_to_dict_conversion()
    test_from_dict_conversion()
    test_metadata_enrichment()
    test_stage3_fields_available()
    
    print("\n" + "="*70)
    print("✅ ALL RESULT TESTS PASSED")
    print("="*70)
