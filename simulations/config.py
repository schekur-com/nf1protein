"""
Configuration module for simulations.
Centralized location for all magic numbers and tunable parameters.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationConfig:
    """
    Central configuration for Langevin-based occupancy simulations.
    
    All numeric constants extracted from hardcoded values to enable
    systematic parameter exploration and reproducibility.
    """
    
    # ==================== Integration Parameters ====================
    total_time: float = 50.0  # Total simulation time (dimensionless units)
    dt: float = 0.01  # Time step
    
    # Derived quantities
    @property
    def n_steps(self) -> int:
        """Total number of integration steps."""
        return int(self.total_time / self.dt)
    
    # ==================== Structural Parameters ====================
    theta_native: float = 0.65  # Native occupancy (structural baseline)
    
    # ==================== Potential Energy Landscape ====================
    alpha_scale: float = 1.2  # Scaling for angular deviation energy (coupled to omega)
    c1: float = 0.25  # First rugged amplitude
    k1: int = 12  # First rugged wavenumber
    c2: float = 0.18  # Second rugged amplitude
    k2: int = 25  # Second rugged wavenumber
    
    # ==================== Effector Dynamics ====================
    A_effector: float = 10.0  # Effector amplification factor
    
    # ==================== Binding Kinetics ====================
    k_on: float = 0.04  # Binding rate constant
    k_off: float = 0.02  # Unbinding rate constant
    
    # ==================== Colored Noise Parameters ====================
    tau_memory: float = 0.8  # Memory timescale (Ornstein-Uhlenbeck correlation time)
    sigma_noise: float = 0.25  # Noise amplitude
    
    # ==================== Homeostasis Metric ====================
    homeostasis_target: float = 0.5  # Target occupancy for homeostasis (PRE-STAGE-2-BUG)
    homeostasis_tolerance: float = 0.20  # Tolerance around target
    
    # ==================== Stability Metric ====================
    lyapunov_sigmoid_scale: float = 25.0  # Scale factor in sigmoid(-scale * lambda)
    
    # ==================== Gradient Normalization ====================
    gradient_norm_enabled: bool = True  # Enable gradient normalization
    
    # ==================== Noise Scaling ====================
    noise_base: float = 0.6  # Base noise scale
    noise_omega_factor: float = 0.4  # Additional scaling factor for noise (coupled to omega)
    
    # ==================== Violation Threshold ====================
    violation_base: float = 0.25  # Base violation threshold
    violation_omega_factor: float = 0.15  # Additional threshold scaling (coupled to omega)
    
    # ==================== Clipping Bounds ====================
    theta_min: float = 0.01  # Minimum occupancy
    theta_max: float = 0.99  # Maximum occupancy
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dt <= 0 or self.total_time <= 0:
            raise ValueError("dt and total_time must be positive")
        if not (0 < self.theta_native < 1):
            raise ValueError("theta_native must be in (0, 1)")
        if not (0 <= self.homeostasis_tolerance < 1):
            raise ValueError("homeostasis_tolerance must be in [0, 1)")


@dataclass
class ScreeningConfig:
    """Configuration for probabilistic mutation screening."""
    
    iterations: int = 100  # Number of samples per mutation
    debug: bool = False  # Enable debug output
    simulation: SimulationConfig = None  # Simulation parameters
    
    def __post_init__(self):
        """Initialize with default simulation config if not provided."""
        if self.simulation is None:
            self.simulation = SimulationConfig()


# Default instances
DEFAULT_SIMULATION_CONFIG = SimulationConfig()
DEFAULT_SCREENING_CONFIG = ScreeningConfig()
