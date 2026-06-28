# 🚀 NF1-Smart-Redirector-Model: Development Roadmap

**Current Status:** TRL-2 (Academic Sandbox) | **Focus:** Computational Verification & Hypothesis Generation

---

## ✅ STAGE 1: Code Quality Foundation (IN PROGRESS)

### Completed
- ✅ **experimental_calibration.py**: Fixed invalid escape sequence (`\p` → raw string)
- ✅ **requirements.txt**: Added ViennaRNA dependency
- ✅ **pyproject.toml**: Added optional dependencies (`[rna]`, `[dev]`)
- ✅ **mutation_robustness_screen.py**: Implemented `DEBUG` flag for development logging
- ✅ **.gitignore**: Created with selective filtering (preserves `.cif` structures, excludes large JSON metadata)
- ✅ **File Organization**: Moved `statistical_profile_v2.py` to `analysis/` (it's analysis, not unit test)

### Status
- **Syntax Check**: ✅ PASSING
- **Packaging**: ✅ READY
- **Imports**: ✅ VALIDATED
- **Dependencies**: ✅ DOCUMENTED

### Next: Verification
```bash
# Full syntax check
python3 -m py_compile $(find . -name "*.py" -type f)

# Run existing unit tests
python3 -m pytest tests/ -v

# Commit changes
git add -A
git commit -m "Stage 1: Fix syntax, packaging, and dependencies (preserve scientific state)"
```

---

## 🔄 STAGE 2: Code Quality + Scientific Bug Fixes (9 Critical Items)

### ⚠️ CRITICAL BUG FIXES (Colab Evidence-Based)

#### 2.0 **Homeostasis Metric - Complete Redesign** 🔴
From Colab: System exhibits true bistability (0.01 ≈ 0.46, 0.99 ≈ 0.41, middle ≈ 0.12)

This is NOT a simple rename. THREE separate problems:

##### 2.0.A: **Target Definition Wrong**
**Current Problem**:
```python
target = 0.5  # or theta_native
# Assumes single attractor!
```

**Fix**:
```python
# No single target. System has TWO stable basins.
LOW_BASIN_BOUNDARY = 0.25
HIGH_BASIN_BOUNDARY = 0.75
# Middle region [0.25, 0.75] is unstable/transition
```

**Implication**: Homeostasis now means "confined to either basin", not "near 0.5"

##### 2.0.B: **Metric Definition Wrong**
**Current Problem**:
```python
metric = |theta - target|  # Distance-based
# Treats occupancy as 1D position
```

**Fix**:
```python
def compute_occupancy_statistics(trajectory):
    """
    Analyzes actual system state: basin occupancy, transitions, dwell.
    """
    low_basin = np.sum(trajectory < 0.25) / len(trajectory)
    high_basin = np.sum(trajectory > 0.75) / len(trajectory)
    unstable = 1.0 - (low_basin + high_basin)
    
    # Detect transitions (crossings of 0.5)
    crossings = np.sum(np.abs(np.diff(np.sign(trajectory - 0.5)))) / 2
    transition_rate = crossings / len(trajectory)
    
    return {
        "low_occupancy": low_basin,
        "high_occupancy": high_basin,
        "unstable_frac": unstable,
        "transition_rate": transition_rate,
        "dwell_time_mean": compute_dwell_time(trajectory),
    }
```

**Implication**: Metrics now reflect actual bistable dynamics

##### 2.0.C: **Score Type Wrong**
**Current Problem**:
```python
p_homeostasis = 0.0 or 1.0  # Binary
# Loses information about basin strength
```

**Fix**:
```python
def compute_homeostasis_score(occupancy_stats):
    """
    Continuous score reflecting confinement strength.
    """
    # Strong confinement: high occupancy in either basin
    confinement = occupancy_stats["low_occupancy"] + occupancy_stats["high_occupancy"]
    
    # Stable dynamics: low transition rate
    stability = 1.0 - occupancy_stats["transition_rate"]
    
    # Combined homeostasis score [0, 1]
    score = 0.6 * confinement + 0.4 * stability
    return score
```

**Implication**: R-score now reflects bistability strength, not false certainty

**Outcome of 2.0**:
- New function: `compute_occupancy_statistics(trajectory)` → dict
- New function: `compute_homeostasis_score(occupancy_stats)` → float [0,1]
- Remove: `compute_homeostasis_probability()`
- Impact: R-score values change, but now scientifically justified

---

#### 2.1 **Descent Metric - Rename + Foundation for Real Lyapunov** 🟡
**Current Problem**: `lambda_max` suggests Lyapunov, but it's mean gradient

**Stage 2 Task** (Rename + Prepare):
```python
# RENAME current metric
mean_descent_rate = np.mean(gradient_magnitudes)  # NOT Lyapunov!

def compute_descent_metric(trajectory, dt=0.001):
    """
    Mean gradient magnitude (descent velocity).
    This is NOT Lyapunov exponent - see Stage 3 for proper formulation.
    """
    gradients = np.diff(trajectory) / dt
    return np.mean(np.abs(gradients))
```

**Stage 2 Preparation** (for Stage 3):
```python
# Add skeleton for real Lyapunov (Stage 3 work)
def compute_lyapunov_exponent(trajectory, dt, method='qr'):
    """
    PLACEHOLDER: Real Lyapunov computation (Stage 3).
    Requires:
    - Jacobian reconstruction from trajectory
    - QR decomposition or Wolf method
    - Finite-time exponents
    
    Current: Not implemented (Stage 3 task)
    See: DEVELOPMENT_ROADMAP.md Stage 3.2
    """
    raise NotImplementedError("Lyapunov exponent: Stage 3 task")
```

**Implication**: 
- Stage 2 clarifies what's actually being computed
- Stage 3 can add real Lyapunov without confusion

---

#### 2.2 **Configuration Management - Modular Config Classes** 🟡
**Problem**: All parameters in one dict, hard to maintain

**Solution**: Separate domain-specific config classes
```python
# simulations/config.py

class SimulationConfig:
    """Global metadata"""
    VERSION = "2.1.0"
    RANDOM_SEED = 42
    DEBUG = True

class NoiseConfig:
    """Noise/stochasticity parameters"""
    SIGMA = 0.15          # Noise amplitude
    CORR_TIME = 2.0       # Correlation timescale
    NOISE_TYPE = "gaussian"  # or "colored", "lévy"

class LandscapeConfig:
    """ODE parameters"""
    T = 100.0             # Total time
    dt = 0.001            # Time step
    tau = 10.0            # Slow timescale
    alpha = 0.6           # Coupling

class BindingConfig:
    """Binding dynamics"""
    k_on = 0.25           # Binding rate
    k_off = 0.18          # Unbinding rate
    OCCUPANCY_LOW = 0.25
    OCCUPANCY_HIGH = 0.75
    HADDOCK_INFLUENCE = 12.0

class OutputConfig:
    """Output/analysis parameters"""
    SAVE_TRAJECTORY = True
    SAVE_FIGURES = True
    VERBOSE = True
    OUTPUT_DIR = "./results"

# Usage in Stage 2.2
SimulationConfig.VERSION = "2.1.0"
```

**Impact**: Easy to extend, modify, and version control parameters

---

#### 2.3 **RNG Standardization - Single Entropy Source** 🟡
**Problem**: Mixed `np.random.XXX()` calls = non-reproducible

**Solution**: Centralized Generator
```python
# simulations/random.py
import numpy as np
from simulations.config import SimulationConfig

# SINGLE source of randomness
rng = np.random.default_rng(seed=SimulationConfig.RANDOM_SEED)

# Use everywhere:
# ❌ OLD: np.random.normal(0, 1, 100)
# ✅ NEW: rng.normal(0, 1, 100)

# ❌ OLD: np.random.random()
# ✅ NEW: rng.random()
```

**Files to update**:
- `simulations/colored_noise_langevin_model.py`
- `simulations/mutation_robustness_screen.py`
- `simulations/cif_coordinate_bridge.py`

**Impact**: Reproducibility guaranteed, even with parallel runs (different seeds)

---

#### 2.4 **Cache System - Comprehensive State Hash** 🟡
**Problem**: Current key `(omega, haddock)` incomplete
```python
cache_key = (omega, haddock)  # Missing: config, seed, version!
```

**Solution**: Simulation State Hash
```python
import hashlib
import json

class SimulationHash:
    """Create deterministic cache key from ALL state parameters"""
    
    @staticmethod
    def compute(omega, haddock, theta_native, config_version, seed):
        """
        Hash all parameters affecting output.
        """
        state = {
            "omega": float(omega),
            "haddock": float(haddock),
            "theta_native": float(theta_native),
            "config_version": str(config_version),
            "seed": int(seed),
            # Add config hash
            "noise_sigma": float(NoiseConfig.SIGMA),
            "landscape_dt": float(LandscapeConfig.dt),
            "binding_k_on": float(BindingConfig.k_on),
        }
        state_json = json.dumps(state, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()[:16]

# Usage
cache_key = SimulationHash.compute(
    omega=omega_mut,
    haddock=haddock_score,
    theta_native=0.47,
    config_version=SimulationConfig.VERSION,
    seed=rng.integers(0, 2**32)
)
```

**Impact**: Cache stays valid only when ALL relevant parameters unchanged

---

#### 2.5 **solve_sde() Return Enrichment - Complete State** 🟡
**Current** (insufficient):
```python
trajectory, lambda_max = solve_sde(...)
```

**Target** (comprehensive):
```python
result = solve_sde(omega_mut=omega, haddock_score=score, config=Config)

# Returns:
{
    "trajectory": np.ndarray,                # Full θ(t)
    "theta_native": float,                   # Equilibrium
    "violations": int,                       # Constraint breaks
    "descent_metric": float,                 # Mean gradient
    "occupancy": {
        "low_basin": float,                  # θ < 0.25
        "high_basin": float,                 # θ > 0.75
        "unstable": float,                   # 0.25 ≤ θ ≤ 0.75
        "transition_rate": float,            # Crossings/dt
        "dwell_time": float,                 # Mean residence
    },
    "statistics": {
        "mean": float,
        "std": float,
        "min": float,
        "max": float,
        "skewness": float,
        "kurtosis": float,
    },
    "binding_dynamics": {
        "max_occupancy": float,
        "min_occupancy": float,
        "occupancy_change": float,
    },
    "simulation_state": {
        "cache_key": str,
        "config_version": str,
        "seed": int,
        "theta_native": float,
    }
}
```

**Impact**: Stage 3 doesn't need to recompute metrics

---

#### 2.6 **Pipeline Decomposition - 8 Focused Functions** 🟡
**Current**: `run_langevin_simulation_pipeline()` = 180 lines

**Target**: Break into focused components (20-40 lines each)

```python
def load_structure_and_validate(cif_file, config):
    """Load AlphaFold CIF, validate geometry"""
    structure = parse_cif(cif_file)
    validate_chains(structure)
    return structure

def compute_native_occupancy(structure, config):
    """Compute θ_native from structure"""
    theta_native = estimate_occupancy_from_structure(structure)
    return theta_native

def initialize_parameters(omega_mut, haddock_score, config):
    """Set up initial state and parameters"""
    params = {
        "omega": omega_mut,
        "haddock": haddock_score,
        "theta_0": compute_native_occupancy(...),
    }
    return params

def generate_binding_process(duration, sigma, seed, config):
    """Create stochastic noise trajectory"""
    rng = np.random.default_rng(seed)
    noise = generate_colored_noise(duration, sigma, rng, config)
    return noise

def integrate_langevin(initial_state, binding_process, config):
    """Run ODE integration"""
    trajectory = solve_ode_rk45(
        initial_state,
        binding_process,
        config.LandscapeConfig.dt,
        config.LandscapeConfig.T
    )
    return trajectory

def compute_metrics(trajectory, config):
    """Analyze trajectory for occupancy, transitions, etc."""
    metrics = {
        "descent": compute_descent_metric(trajectory, config.dt),
        "occupancy": compute_occupancy_statistics(trajectory),
        "stats": compute_trajectory_statistics(trajectory),
    }
    return metrics

def render_diagnostic_figures(trajectory, metrics, config):
    """Generate visualization"""
    if config.OutputConfig.SAVE_FIGURES:
        plot_trajectory(trajectory)
        plot_occupancy(metrics["occupancy"])
    return None

def build_result_object(trajectory, metrics, config, cache_key):
    """Assemble final output dict"""
    result = {
        "trajectory": trajectory,
        **metrics,
        "simulation_state": {"cache_key": cache_key, "config": config},
    }
    return result

def run_langevin_simulation_pipeline(cif_file, omega_mut, haddock_score, config):
    """Orchestrate: load → compute → integrate → analyze"""
    structure = load_structure_and_validate(cif_file, config)
    params = initialize_parameters(omega_mut, haddock_score, config)
    noise = generate_binding_process(config.LandscapeConfig.T, 
                                    config.NoiseConfig.SIGMA, 
                                    rng.integers(0, 2**32), 
                                    config)
    trajectory = integrate_langevin(params["theta_0"], noise, config)
    metrics = compute_metrics(trajectory, config)
    cache_key = SimulationHash.compute(omega_mut, haddock_score, 
                                       params["theta_0"], 
                                       config.SimulationConfig.VERSION,
                                       rng.integers(0, 2**32))
    render_diagnostic_figures(trajectory, metrics, config)
    result = build_result_object(trajectory, metrics, config, cache_key)
    return result
```

**Impact**: Each function testable, replaceable, clear responsibility

---

#### 2.7 **Trajectory Analysis Module - Attractor Detection** 🟡
**NEW**: Colab showed us system transitions 0.01 ↔ 0.99

**Create**: `simulations/trajectory_analysis.py`

```python
def detect_attractors(trajectory, bins=20):
    """
    Identify stable states from trajectory histogram
    """
    counts, bin_edges = np.histogram(trajectory, bins=bins)
    peaks = detect_peaks(counts)
    attractors = bin_edges[peaks]
    return attractors

def detect_transitions(trajectory, threshold=0.5):
    """
    Find crossings of threshold (state changes)
    """
    crossings = np.where(np.diff(np.sign(trajectory - threshold)))[0]
    return crossings

def estimate_dwell_times(trajectory, attractors):
    """
    How long in each basin before transition?
    """
    dwell_times = {}
    for attractor in attractors:
        basin = np.where(np.abs(trajectory - attractor) < 0.1)[0]
        # Find continuous segments
        segments = np.split(basin, np.where(np.diff(basin) > 1)[0])
        dwell_times[attractor] = [len(seg) for seg in segments]
    return dwell_times

def estimate_occupancy_and_entropy(trajectory):
    """
    Basin occupancy and transition entropy
    """
    occupancy = {
        "low": np.sum(trajectory < 0.25) / len(trajectory),
        "high": np.sum(trajectory > 0.75) / len(trajectory),
        "mid": np.sum((trajectory >= 0.25) & (trajectory <= 0.75)) / len(trajectory),
    }
    
    # Shannon entropy
    p = np.array(list(occupancy.values()))
    entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    
    return occupancy, entropy
```

**Impact**: Stage 3 can directly use these for bistability characterization

---

#### 2.8 **Testing - Golden Tests & Regression** 🟡
**Create**: `tests/test_refactoring_regression.py`

```python
def test_old_vs_new_deterministic():
    """
    Golden test: refactored code produces identical output
    with same seed as original.
    """
    seed = 42
    omega, haddock = 0.5, 0.3
    
    # Run refactored code
    result_new = run_langevin_simulation_pipeline(
        cif_file="alphafold_models/fold_2026_05_15_18_38_model_0.cif",
        omega_mut=omega,
        haddock_score=haddock,
        config=Config
    )
    
    # Compare against stored baseline (from Stage 1)
    baseline = np.load("tests/golden_baseline.npy")
    
    # Trajectories should be bitwise identical (or nearly identical within eps)
    assert np.allclose(result_new["trajectory"], baseline, rtol=1e-10)

def test_rng_reproducibility():
    """RNG with same seed produces same random numbers"""
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)
    
    assert np.array_equal(rng1.normal(0, 1, 100), 
                         rng2.normal(0, 1, 100))

def test_config_version_isolation():
    """Changing config changes cache key"""
    hash1 = SimulationHash.compute(0.5, 0.3, 0.47, "2.0.0", 42)
    
    SimulationConfig.VERSION = "2.1.0"
    hash2 = SimulationHash.compute(0.5, 0.3, 0.47, "2.1.0", 42)
    
    assert hash1 != hash2
```

**Impact**: Refactoring validated; prevents silent regressions

---

#### 2.9 **Integration Test - End-to-End Mutation Screening** 🟡
**Create**: `tests/test_mutation_screening_e2e.py`

```python
def test_mutation_screening_with_new_metrics():
    """
    Full pipeline with new homeostasis metric.
    Verify R-scores are now continuous and scientifically sensible.
    """
    results = run_probabilistic_screening(iterations=10)
    
    for mut_name, score in results.items():
        # R-score should be [0, 1]
        assert 0 <= score["r_score"] <= 1
        
        # Occupancy should sum to ~1
        occ_sum = (score["occupancy"]["low_basin"] + 
                   score["occupancy"]["high_basin"] + 
                   score["occupancy"]["unstable"])
        assert np.isclose(occ_sum, 1.0, rtol=1e-3)
        
        # Transition rate should be [0, 1] or make sense
        assert 0 <= score["occupancy"]["transition_rate"]
```

**Impact**: Confidence that entire pipeline works with new metrics

---

---

### Code Quality Refactoring (Do NOT change scientific behavior)

**Note**: Items 2.2-2.9 detailed above. These are the critical refactoring tasks that make Stage 3 possible.

---

## 🧪 STAGE 3: Scientific Validation & Hypothesis Testing

### After Code Quality Foundation (Stage 2 Complete)

Once Stage 2 provides clean, testable infrastructure:

#### 3.1 **Bistability Phase Portrait Analysis**
**Use Stage 2.7-2.8 outputs**:
- Compute 2D phase portraits with actual attractors and separatrix
- Measure basin volumes (probability density, not just occupancy %)
- Escape probability from each basin
- Parameter sensitivity (omega → basin structure)

#### 3.2 **Real Lyapunov Exponent Computation** (Filling 2.1 placeholder)
**Use Stage 2.1 infrastructure**:
- Jacobian reconstruction from trajectory
- Wolf method or QR decomposition
- Local vs global Lyapunov spectrum
- Relationship to descent_metric from Stage 2

#### 3.3 **HADDOCK-Dynamics Coupling Validation**
- Is omega linearly affected by HADDOCK score?
- Non-linear coupling terms?
- Validation against experimental NMR shifts
- Bistability robustness to HADDOCK perturbations

#### 3.4 **Omega Prior Calibration & Sensitivity**
- Beta(6,4) for R1276Q: justified by what?
- Sensitivity analysis: how much does prior choice affect R-score?
- Could maximum entropy priors improve predictions?

#### 3.5 **Extended Bifurcation Analysis**
- HADDOCK score as bifurcation parameter
- Track how basin structure changes with omega
- Critical points (saddle-node, Hopf transitions?)
- Hysteresis effects

---

## 📊 Performance Monitoring

### Caching Metrics
```bash
# Monitor cache hit ratio
export DEBUG=True  # Enables cache logging
python3 -c "from simulations import mutation_robustness_screen; ..."
```

### Profiling
```bash
# Identify bottlenecks
python3 -m cProfile -s cumtime simulations/mutation_robustness_screen.py
```

---

---

## ⚠️ CRITICAL NOTES FOR STAGE 2 EXECUTION

### DO Apply These Changes (Bug Fixes)
- ✅ **Homeostasis metric rewrite** - 0.01/0.99 bistability data-driven
- ✅ **Descent metric renaming** - clarity on what's actually computed
- ✅ Extract magic numbers to config
- ✅ Standardize RNG
- ✅ Extend cache key
- ✅ Decompose pipeline functions

### DO NOT Apply These Changes (Without Testing)
- ❌ Modify ODE solver logic without numerical validation
- ❌ Change integration method (RK45 parameters)
- ❌ Alter beta priors without Stage 3 justification

### These WILL Change Results (Expected)
- 🔄 R-score values (due to new bistability metric)
- 🔄 Mutation rankings (derivative of r-score change)
- 🔄 Console output (renamed metrics)

### These Will NOT Change Results (Code Refactoring)
- 🔄 Numerical outputs should be identical (within floating point precision)
- 🔄 Cache behavior should be equivalent (with improved completeness)
- 🔄 Performance should stay the same or improve

---

## 📋 Checklist

### Stage 1 (Completed ✅)
- [x] Syntax validation
- [x] Dependency documentation
- [x] File organization
- [x] Development logging setup
- [x] Git ready for commit

### Stage 2 (Next - 9 Items: Bug Fixes + Code Quality)
- [ ] **2.0.A: Homeostasis Target** - Single attractor → bistable basins
- [ ] **2.0.B: Homeostasis Metric** - Distance → occupancy/transition analysis
- [ ] **2.0.C: Homeostasis Score** - Binary → continuous [0,1]
- [ ] **2.1: Descent Metric Rename** - Clarity + Lyapunov infrastructure
- [ ] **2.2: Config Classes** - 5 modular domain-specific configs
- [ ] **2.3: RNG Standardization** - Single np.random.default_rng() source
- [ ] **2.4: Cache Extension** - SimulationHash with all parameters
- [ ] **2.5: solve_sde() Enrichment** - Return comprehensive dict
- [ ] **2.6: Pipeline Decomposition** - 8 focused functions (20-40 lines each)
- [ ] **2.7: Trajectory Analysis Module** - Attractors, transitions, dwell times
- [ ] **2.8: Testing** - Golden tests + regression validation
- [ ] **2.9: E2E Integration Test** - Full mutation screening with new metrics

### Stage 3 (Later - Scientific Validation)
- [ ] Bistability phase portrait (use Stage 2.7 outputs)
- [ ] Real Lyapunov computation (use Stage 2.1 infrastructure)
- [ ] HADDOCK coupling validation
- [ ] Omega prior calibration
- [ ] Extended bifurcation analysis

---

---

## ⚠️ CRITICAL ORDERING

**DO NOT skip ahead or reorder!**

```
Stage 2.0 (Homeostasis 3-part bug fix)  ← FOUNDATION
    ↓
Stage 2.1 (Descent clarity + Lyapunov prep)  ← NAMING CLARITY
    ↓
Stage 2.2-2.6 (Config, RNG, Cache, Return, Pipeline)  ← REFACTORING
    ↓
Stage 2.7 (Trajectory analysis)  ← STAGE 3 PREREQUISITES
    ↓
Stage 2.8-2.9 (Testing)  ← VALIDATION
    ↓
Stage 3 (Scientific work)  ← BUILDS ON SOLID FOUNDATION
```

**Why this order?**
1. Fix bugs FIRST (2.0-2.1) before refactoring
2. Config + RNG must exist before testing (2.2-2.3)
3. Trajectory module needed for Stage 3 analysis (2.7)
4. Tests validate entire pipeline works (2.8-2.9)
5. Stage 3 doesn't write code, only does analysis

---

## 📊 Key Metrics to Track During Stage 2

### Golden Test Results
```
SEED=42: OLD_CODE trajectory == NEW_CODE trajectory ?
→ Expected: np.allclose(old, new, rtol=1e-10)
```

### Performance Impact
```
Time solve_sde(omega=0.5, haddock=0.3):
  Before: _____ ms
  After:  _____ ms (should be ≤10% slower from logging)
```

### Config Coverage
```
Magic numbers found: _____ → Extracted: _____ (target: 100%)
```

### Cache Efficiency
```
Cache hits: _____% (target: >70% on repeated runs)
Collisions: _____ (target: 0)
```

---

## 🎯 Expected Outcomes After Stage 2

✅ **Code Quality**:
- All functions have single responsibility
- No magic numbers scattered in code
- RNG is reproducible and testable
- Cache handles all relevant parameters

✅ **Scientific Correctness**:
- Homeostasis metric reflects bistability
- R-score is now continuous, scientifically justified
- Trajectory analysis ready for Stage 3
- Results validation tests pass

✅ **Documentation**:
- Each function has docstring
- Config parameters well-documented
- Trajectory analysis module exported and tested
- Lyapunov framework ready (placeholder + docs)

⚠️ **NOT Changed** (Stage 2 preserves):
- ODE solver logic (no integration method changes)
- Beta priors (kept as-is pending Stage 3)
- Fundamental dynamics (refactoring only)

---

**Last Updated**: 2026-06-28 (FINAL REVISION - 9-item Stage 2 with complete specifications)  
**Status**: Stage 1 Complete ✅ | Stage 2 Ready for Implementation 📋 | Stage 3 Will Follow 📊

**Total Stage 2 Implementation Effort**: ~40-60 hours (refactoring + testing)
**Files to Create**: `simulations/config.py`, `simulations/random.py`, `simulations/trajectory_analysis.py`, `tests/test_refactoring_regression.py`, `tests/test_mutation_screening_e2e.py`
**Files to Modify**: 8+ files in `simulations/`, `bridge_models/`, `notebooks/`
