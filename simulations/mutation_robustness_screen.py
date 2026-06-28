import numpy as np
import scipy.stats as stats

# Debug flag for development
DEBUG = True

from simulations.colored_noise_langevin_model import solve_sde
from simulations.config import SimulationConfig, DEFAULT_SIMULATION_CONFIG
from simulations import random as sim_random
from bridge_models.evidence_weighted_calibration import (
    load_haddock_score_from_json
)

TRUE_SCORE, TRUE_STD = load_haddock_score_from_json()

MUTATION_MANIFOLD = {
    "R1276Q (Missense - GAP Domain)": {
        "evidence": "strong",
        "omega_prior": stats.beta(6, 4),
        "haddock_score": TRUE_SCORE,
        "haddock_std": TRUE_STD
    },
    "R681X (Nonsense - Severe)": {
        "evidence": "strong",
        "omega_prior": stats.beta(1, 18),
        "haddock_score": 0.0,
        "haddock_std": 0.0
    },
    "c.2041C>T (Splice - Exon Skip)": {
        "evidence": "moderate",
        "omega_prior": stats.beta(3, 9),
        "haddock_score": TRUE_SCORE * 0.58,
        "haddock_std": TRUE_STD * 1.5
    }
}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_homeostasis_probability(
    trajectory,
    config=None
):
    """
    Compute homeostasis probability based on trajectory.
    
    Args:
        trajectory: Array of occupancy values
        config: SimulationConfig (uses defaults if None)
        
    Returns:
        float: 1.0 if homeostatic, 0.0 otherwise
    """
    if config is None:
        config = DEFAULT_SIMULATION_CONFIG
    
    trajectory = np.asarray(trajectory)

    if len(trajectory) < 2000:
        window = trajectory
    else:
        window = trajectory[-2000:]

    mean_theta = np.mean(window)

    return float(
        abs(mean_theta - config.homeostasis_target) < config.homeostasis_tolerance
    )

def compute_lambda_component(mean_lambda, config=None):
    """
    Compute stability component from mean lambda.
    
    Negatif Lyapunov = daha stabil (returns higher score)
    Pozitif Lyapunov = daha kaotik (returns lower score)
    
    Args:
        mean_lambda: Mean descent metric value
        config: SimulationConfig (uses defaults if None)
        
    Returns:
        float: Sigmoid-transformed stability score [0, 1]
    """
    if config is None:
        config = DEFAULT_SIMULATION_CONFIG
    
    return sigmoid(-config.lyapunov_sigmoid_scale * mean_lambda)

def _sample_mutation_parameters(mutation_meta, iterations):
    """Sample omega and haddock_score for a given mutation.
    
    Args:
        mutation_meta: Dict with 'omega_prior', 'haddock_score', 'haddock_std'
        iterations: Number of samples
        
    Returns:
        tuple: (sampled_omegas, sampled_scores)
    """
    sampled_omegas = mutation_meta["omega_prior"].rvs(iterations)
    
    if mutation_meta["haddock_std"] > 0:
        sampled_scores = sim_random.normal(
            loc=mutation_meta["haddock_score"],
            scale=mutation_meta["haddock_std"],
            size=iterations
        )
    else:
        sampled_scores = np.zeros(iterations)
    
    return sampled_omegas, sampled_scores

def _run_single_trajectory(omega, haddock_score, config=None):
    """Run one SDE integration.
    
    Args:
        omega: Mutation parameter
        haddock_score: Binding affinity perturbation
        config: SimulationConfig (uses defaults if None)
        
    Returns:
        tuple: (trajectory, lambda_max)
    """
    if config is None:
        config = DEFAULT_SIMULATION_CONFIG
    
    trajectory, lambda_max = solve_sde(
        omega_mut=float(omega),
        haddock_score=float(haddock_score),
        config=config
    )
    return trajectory, lambda_max

def _accumulate_trajectory_statistics(sampled_omegas, sampled_scores, config=None):
    """Run all trajectories for a mutation and accumulate statistics.
    
    Args:
        sampled_omegas: Array of omega samples
        sampled_scores: Array of haddock_score samples
        config: SimulationConfig (uses defaults if None)
        
    Returns:
        tuple: (homeostasis_hits, lyapunov_pool)
    """
    if config is None:
        config = DEFAULT_SIMULATION_CONFIG
    
    homeostasis_hits = 0
    lyapunov_pool = []
    
    for omega, score in zip(sampled_omegas, sampled_scores):
        trajectory, lambda_max = _run_single_trajectory(omega, score, config)
        
        lyapunov_pool.append(float(lambda_max))
        
        homeostasis_hits += compute_homeostasis_probability(
            trajectory,
            config=config
        )
    
    return homeostasis_hits, lyapunov_pool

def _compute_rescue_metrics(homeostasis_hits, lyapunov_pool, iterations, config=None):
    """Compute aggregate metrics from trajectory statistics.
    
    Args:
        homeostasis_hits: Count of trajectories meeting homeostasis criterion
        lyapunov_pool: List of lambda_max values
        iterations: Total number of simulations
        config: SimulationConfig (uses defaults if None)
        
    Returns:
        dict: {'p_homeostasis', 'mean_lambda', 'lambda_component', 'r_score'}
    """
    if config is None:
        config = DEFAULT_SIMULATION_CONFIG
    
    p_homeostasis = homeostasis_hits / float(iterations)
    mean_lambda = float(np.mean(lyapunov_pool))
    lambda_component = compute_lambda_component(mean_lambda, config=config)
    
    r_score = (
        0.5 * lambda_component +
        0.5 * p_homeostasis
    )
    
    return {
        'p_homeostasis': p_homeostasis,
        'mean_lambda': mean_lambda,
        'lambda_component': lambda_component,
        'r_score': r_score,
        'lyapunov_pool': lyapunov_pool  # Keep for debugging
    }

def _print_mutation_results(mut_name, mutation_meta, metrics):
    """Print results for one mutation.
    
    Args:
        mut_name: Mutation name
        mutation_meta: Metadata dict with 'evidence'
        metrics: Dict from _compute_rescue_metrics()
    """
    print()
    if DEBUG:
        print("min lambda =", np.min(metrics['lyapunov_pool']))
        print("mean lambda =", metrics['mean_lambda'])
        print("max lambda =", np.max(metrics['lyapunov_pool']))
        print("lambda_component =", metrics['lambda_component'])
        print("p_homeostasis =", metrics['p_homeostasis'])
    
    print(
        f"▶ Varyant: {mut_name} "
        f"[Kanit Duzeyi: {mutation_meta['evidence'].upper()}]"
    )
    print(
        f"  └─ Olasiliksal Kurtarma P(homeostasis): "
        f"%{100.0 * metrics['p_homeostasis']:.2f}"
    )
    print(
        f"  └─ Ortalama Lyapunov Kararliligi: "
        f"{metrics['mean_lambda']:.4f}"
    )
    print(
        f"  └─ Phenotypic Dynamical Rescue Index (R): "
        f"{metrics['r_score']:.4f}"
    )

def _run_screening_for_mutation(mut_name, mutation_meta, iterations, config=None):
    """Orchestrate full analysis for one mutation.
    
    Args:
        mut_name: Mutation name
        mutation_meta: Metadata dict
        iterations: Number of replicates
        config: SimulationConfig (uses defaults if None)
        
    Returns:
        dict: Computed metrics including 'r_score'
    """
    if config is None:
        config = DEFAULT_SIMULATION_CONFIG
    
    sampled_omegas, sampled_scores = _sample_mutation_parameters(
        mutation_meta, iterations
    )
    
    homeostasis_hits, lyapunov_pool = _accumulate_trajectory_statistics(
        sampled_omegas, sampled_scores, config=config
    )
    
    metrics = _compute_rescue_metrics(homeostasis_hits, lyapunov_pool, iterations, config=config)
    
    _print_mutation_results(mut_name, mutation_meta, metrics)
    
    return metrics

def run_probabilistic_screening(iterations=100, config=None):
    """
    Main entry point for mutation screening pipeline.
    
    Args:
        iterations: Number of samples per mutation
        config: SimulationConfig (uses defaults if None)
    """
    if config is None:
        config = DEFAULT_SIMULATION_CONFIG
    
    print("🔬 RUNNING: Data-Driven Mutation-Aware Pathology Manifold")
    print("=" * 65)
    
    for mut_name, meta in MUTATION_MANIFOLD.items():
        _run_screening_for_mutation(mut_name, meta, iterations, config=config)

if __name__ == "__main__":
    run_probabilistic_screening()
