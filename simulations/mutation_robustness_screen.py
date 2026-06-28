import numpy as np
import scipy.stats as stats

# Debug flag for development
DEBUG = True

from simulations.colored_noise_langevin_model import solve_sde
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
    target=0.5,
    tolerance=0.20
):
    trajectory = np.asarray(trajectory)

    if len(trajectory) < 2000:
        window = trajectory
    else:
        window = trajectory[-2000:]

    mean_theta = np.mean(window)

    return float(
        abs(mean_theta - target) < tolerance
    )

def compute_lambda_component(mean_lambda):
    """
    Negatif Lyapunov = daha stabil
    Pozitif Lyapunov = daha kaotik
    """
    return sigmoid(-25.0 * mean_lambda)

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
        sampled_scores = np.random.normal(
            loc=mutation_meta["haddock_score"],
            scale=mutation_meta["haddock_std"],
            size=iterations
        )
    else:
        sampled_scores = np.zeros(iterations)
    
    return sampled_omegas, sampled_scores

def _run_single_trajectory(omega, haddock_score):
    """Run one SDE integration.
    
    Args:
        omega: Mutation parameter
        haddock_score: Binding affinity perturbation
        
    Returns:
        tuple: (trajectory, lambda_max)
    """
    trajectory, lambda_max = solve_sde(
        omega_mut=float(omega),
        haddock_score=float(haddock_score)
    )
    return trajectory, lambda_max

def _accumulate_trajectory_statistics(sampled_omegas, sampled_scores):
    """Run all trajectories for a mutation and accumulate statistics.
    
    Args:
        sampled_omegas: Array of omega samples
        sampled_scores: Array of haddock_score samples
        
    Returns:
        tuple: (homeostasis_hits, lyapunov_pool)
    """
    homeostasis_hits = 0
    lyapunov_pool = []
    
    for omega, score in zip(sampled_omegas, sampled_scores):
        trajectory, lambda_max = _run_single_trajectory(omega, score)
        
        lyapunov_pool.append(float(lambda_max))
        
        homeostasis_hits += compute_homeostasis_probability(
            trajectory,
            target=0.5,
            tolerance=0.20
        )
    
    return homeostasis_hits, lyapunov_pool

def _compute_rescue_metrics(homeostasis_hits, lyapunov_pool, iterations):
    """Compute aggregate metrics from trajectory statistics.
    
    Args:
        homeostasis_hits: Count of trajectories meeting homeostasis criterion
        lyapunov_pool: List of lambda_max values
        iterations: Total number of simulations
        
    Returns:
        dict: {'p_homeostasis', 'mean_lambda', 'lambda_component', 'r_score'}
    """
    p_homeostasis = homeostasis_hits / float(iterations)
    mean_lambda = float(np.mean(lyapunov_pool))
    lambda_component = compute_lambda_component(mean_lambda)
    
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

def _run_screening_for_mutation(mut_name, mutation_meta, iterations):
    """Orchestrate full analysis for one mutation.
    
    Args:
        mut_name: Mutation name
        mutation_meta: Metadata dict
        iterations: Number of replicates
        
    Returns:
        dict: Computed metrics including 'r_score'
    """
    sampled_omegas, sampled_scores = _sample_mutation_parameters(
        mutation_meta, iterations
    )
    
    homeostasis_hits, lyapunov_pool = _accumulate_trajectory_statistics(
        sampled_omegas, sampled_scores
    )
    
    metrics = _compute_rescue_metrics(homeostasis_hits, lyapunov_pool, iterations)
    
    _print_mutation_results(mut_name, mutation_meta, metrics)
    
    return metrics

def run_probabilistic_screening(iterations=100):
    """Main entry point for mutation screening pipeline."""
    print("🔬 RUNNING: Data-Driven Mutation-Aware Pathology Manifold")
    print("=" * 65)
    
    for mut_name, meta in MUTATION_MANIFOLD.items():
        _run_screening_for_mutation(mut_name, meta, iterations)

if __name__ == "__main__":
    run_probabilistic_screening()
