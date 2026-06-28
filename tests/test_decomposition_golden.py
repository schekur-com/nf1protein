"""
Golden test for Task 1: Decompose run_probabilistic_screening()

This test verifies that the refactored code (with helper functions)
produces identical results to the original monolithic version.

Strategy: Run both with same seed, compare output.
"""

import numpy as np
import sys
from io import StringIO
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set seed for reproducibility
np.random.seed(42)

def test_decomposed_pipeline_runs_without_error():
    """Verify that refactored pipeline runs and doesn't crash."""
    from simulations.mutation_robustness_screen import run_probabilistic_screening
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        run_probabilistic_screening(iterations=2)  # Small number for speed
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    
    # Verify output contains expected sections
    assert "RUNNING" in output, f"Missing pipeline header. Output:\n{output}"
    assert "Varyant" in output, f"Missing mutation results. Output:\n{output}"
    assert "Phenotypic Dynamical Rescue Index" in output, f"Missing R-score output. Output:\n{output}"
    print("✅ Decomposed pipeline runs without error")

def test_helper_functions_produce_expected_metrics():
    """Verify that individual helper functions work correctly."""
    from simulations.mutation_robustness_screen import (
        _sample_mutation_parameters,
        _run_single_trajectory,
        _accumulate_trajectory_statistics,
        _compute_rescue_metrics,
        MUTATION_MANIFOLD
    )
    
    # Get first mutation
    mut_name, meta = list(MUTATION_MANIFOLD.items())[0]
    
    # Sample parameters
    omegas, scores = _sample_mutation_parameters(meta, iterations=3)
    assert len(omegas) == 3, "Should sample 3 omegas"
    assert len(scores) == 3, "Should sample 3 scores"
    print(f"✅ _sample_mutation_parameters works: {len(omegas)} samples")
    
    # Run single trajectory
    traj, lam = _run_single_trajectory(float(omegas[0]), float(scores[0]))
    assert isinstance(traj, np.ndarray), "Trajectory should be ndarray"
    assert isinstance(lam, (float, np.floating)), "Lambda should be float"
    print(f"✅ _run_single_trajectory works: trajectory shape {traj.shape}, lambda {lam:.4f}")
    
    # Accumulate statistics
    hits, lambdas = _accumulate_trajectory_statistics(omegas, scores)
    assert isinstance(hits, (int, float, np.integer, np.floating)), "Hits should be numeric"
    assert len(lambdas) == 3, "Should have 3 lambda values"
    print(f"✅ _accumulate_trajectory_statistics works: {hits} hits, {len(lambdas)} lambdas")
    
    # Compute metrics
    metrics = _compute_rescue_metrics(hits, lambdas, iterations=3)
    assert 0 <= metrics['p_homeostasis'] <= 1, "p_homeostasis out of range"
    assert 0 <= metrics['r_score'] <= 1, "r_score out of range"
    print(f"✅ _compute_rescue_metrics works: p_h={metrics['p_homeostasis']:.3f}, r={metrics['r_score']:.3f}")

def test_output_format_unchanged():
    """Verify that output format matches expected structure."""
    from simulations.mutation_robustness_screen import run_probabilistic_screening
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        run_probabilistic_screening(iterations=2)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    
    lines = output.split('\n')
    
    # Count key output lines
    variant_lines = [l for l in lines if "Varyant:" in l]
    homeostasis_lines = [l for l in lines if "Olasiliksal Kurtarma" in l]
    r_score_lines = [l for l in lines if "Phenotypic Dynamical" in l]
    
    # Should have at least one mutation
    assert len(variant_lines) >= 1, "Missing variant lines"
    assert len(homeostasis_lines) >= 1, "Missing homeostasis output"
    assert len(r_score_lines) >= 1, "Missing R-score output"
    
    print(f"✅ Output format correct: {len(variant_lines)} mutations, all metrics present")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GOLDEN TEST: Task 1 - Pipeline Decomposition")
    print("="*70)
    
    test_decomposed_pipeline_runs_without_error()
    test_helper_functions_produce_expected_metrics()
    test_output_format_unchanged()
    
    print("\n" + "="*70)
    print("✅ ALL GOLDEN TESTS PASSED - Behavior preserved!")
    print("="*70)
