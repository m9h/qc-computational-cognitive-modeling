"""Autoresearch infrastructure — READ ONLY.

Provides solver wrappers, metric computation, and result logging for:
1. Consensus time experiments (opinion dynamics on frustrated networks)
2. Quantum minority game experiments (quantum strategy selection)
3. Ground state search experiments (spin glass energy optimization)
"""

from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import dataclass, field
from itertools import product as iproduct
from pathlib import Path

import numpy as np

from qcccm.spin_glass.hamiltonians import (
    SocialSpinGlassParams,
    ea_couplings,
    frustration_index,
    sk_couplings,
    social_hamiltonian_classical,
)
from qcccm.spin_glass.order_params import (
    binder_cumulant,
    edwards_anderson_q,
    glass_susceptibility,
    overlap,
    overlap_distribution,
)
from qcccm.spin_glass.solvers import (
    SolverResult,
    metropolis_spin_glass,
    transverse_field_mc,
    vqe_ground_state,
    qaoa_ground_state,
)
from qcccm.games.minority import (
    MinorityGameParams,
    GameHistory,
    run_minority_game,
    volatility_sweep,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_FILE = Path(__file__).parent / "results.tsv"
LOG_FILE = Path(__file__).parent / "run.log"
TIMEOUT_SECONDS = 600
BRUTE_FORCE_MAX_N = 20


# ---------------------------------------------------------------------------
# Exact ground state (brute force)
# ---------------------------------------------------------------------------


def exact_ground_state(params: SocialSpinGlassParams) -> tuple[float, np.ndarray]:
    """Find exact ground state by exhaustive enumeration."""
    N = params.n_agents
    if N > BRUTE_FORCE_MAX_N:
        raise ValueError(f"N={N} too large for brute force (max {BRUTE_FORCE_MAX_N})")
    best_E = float("inf")
    best_s = None
    for bits in iproduct([-1, 1], repeat=N):
        s = np.array(bits, dtype=float)
        E = social_hamiltonian_classical(params, s)
        if E < best_E:
            best_E = E
            best_s = s.copy()
    return best_E, best_s


# ---------------------------------------------------------------------------
# Consensus time measurement
# ---------------------------------------------------------------------------


def measure_consensus_time(
    params: SocialSpinGlassParams,
    n_steps: int = 10000,
    threshold: float = 0.9,
    use_pimc: bool = False,
    n_trotter: int = 8,
) -> tuple[int, np.ndarray]:
    """Run MC dynamics and measure time to reach consensus.

    Consensus = |magnetization| >= threshold.
    Returns (consensus_time, magnetization_trajectory).
    If consensus not reached, returns (n_steps, trajectory).
    """
    N = params.n_agents
    rng = np.random.default_rng(params.seed)
    spins = rng.choice([-1, 1], size=N).astype(float)
    T = max(params.temperature, 1e-10)
    Gamma = params.transverse_field

    mag_trajectory = []

    if use_pimc and Gamma > 0:
        # PIMC: use Trotter replicas
        P = n_trotter
        J_tau = -0.5 * T * np.log(np.tanh(Gamma / (P * T) + 1e-15)) if Gamma > 0 else 0.0
        replicas = np.tile(spins, (P, 1))

        for step in range(n_steps):
            for _ in range(N * P):
                p = rng.integers(0, P)
                i = rng.integers(0, N)
                delta_E_in = 2.0 * replicas[p, i] * (
                    np.dot(params.J[i], replicas[p]) / P + params.fields[i] / P
                )
                p_prev, p_next = (p - 1) % P, (p + 1) % P
                delta_E_tau = 2.0 * J_tau * replicas[p, i] * (
                    replicas[p_prev, i] + replicas[p_next, i]
                )
                delta_E = delta_E_in + delta_E_tau
                if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                    replicas[p, i] *= -1

            mag = abs(np.mean(replicas[0]))
            mag_trajectory.append(mag)
            if mag >= threshold:
                return step + 1, np.array(mag_trajectory)
    else:
        # Standard Metropolis
        for step in range(n_steps):
            i = rng.integers(0, N)
            delta_E = 2.0 * spins[i] * (np.dot(params.J[i], spins) + params.fields[i])
            if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                spins[i] *= -1

            mag = abs(np.mean(spins))
            mag_trajectory.append(mag)
            if mag >= threshold:
                return step + 1, np.array(mag_trajectory)

    return n_steps, np.array(mag_trajectory)


# ---------------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Container for a single experiment's results."""

    commit: str = ""
    description: str = ""

    # Model specification
    model: str = "SK"
    topology: str = "complete"
    disorder: str = "gaussian"
    n_agents: int = 8
    temperature: float = 0.5
    transverse_field: float = 0.0
    frustration: float = 0.0
    seed: int = 42

    # Results
    method: str = "metropolis"
    E_best: float = 0.0
    E_exact: float = 0.0
    quantum_advantage: float = 0.0
    q_EA: float = 0.0
    wall_time: float = 0.0
    status: str = "ok"

    # Secondary metrics
    magnetization: float = 0.0
    frustration_index: float = 0.0
    binder: float = 0.0

    # Consensus time metrics (new)
    consensus_time: int = 0
    consensus_speedup: float = 0.0  # tau_classical / tau_quantum

    # Minority game metrics (new)
    volatility: float = 0.0
    volatility_reduction: float = 0.0  # (vol_classical - vol_quantum) / vol_classical

    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_quantum_advantage(
    E_classical: float,
    E_quantum: float,
    E_exact: float,
) -> float:
    """Energy-based quantum advantage (positive = quantum better)."""
    if abs(E_exact) < 1e-10:
        return 0.0
    return (E_classical - E_quantum) / abs(E_exact)


def compute_consensus_speedup(
    tau_classical: int,
    tau_quantum: int,
    max_steps: int,
) -> float:
    """Consensus speedup (> 1 means quantum reaches consensus faster).

    Returns 0 if neither reached consensus, tau_classical/tau_quantum otherwise.
    """
    if tau_classical >= max_steps and tau_quantum >= max_steps:
        return 0.0  # neither reached consensus
    if tau_quantum >= max_steps:
        return -1.0  # classical reached but quantum didn't
    if tau_classical >= max_steps:
        return float(max_steps) / tau_quantum  # quantum reached, classical didn't
    return tau_classical / tau_quantum


def compute_volatility_reduction(
    vol_classical: float,
    vol_quantum: float,
) -> float:
    """Volatility reduction in minority game (positive = quantum reduces herding)."""
    if vol_classical < 1e-10:
        return 0.0
    return (vol_classical - vol_quantum) / vol_classical


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------


HEADER = [
    "commit", "model", "topology", "disorder", "N", "T", "Gamma",
    "frustration", "method", "E_best", "E_exact", "quantum_advantage",
    "q_EA", "magnetization", "frustration_index", "binder",
    "consensus_time", "consensus_speedup",
    "volatility", "volatility_reduction",
    "wall_time", "status", "description",
]


def log_result(result: ExperimentResult) -> None:
    """Append result to results.tsv."""
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(HEADER)
        writer.writerow([
            result.commit, result.model, result.topology, result.disorder,
            result.n_agents, f"{result.temperature:.4f}",
            f"{result.transverse_field:.4f}", f"{result.frustration:.4f}",
            result.method, f"{result.E_best:.6f}", f"{result.E_exact:.6f}",
            f"{result.quantum_advantage:.6f}", f"{result.q_EA:.6f}",
            f"{result.magnetization:.6f}", f"{result.frustration_index:.6f}",
            f"{result.binder:.6f}",
            result.consensus_time, f"{result.consensus_speedup:.4f}",
            f"{result.volatility:.6f}", f"{result.volatility_reduction:.6f}",
            f"{result.wall_time:.2f}", result.status, result.description,
        ])


def get_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent, text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Solver runners with timing
# ---------------------------------------------------------------------------


def run_classical(params, n_steps=5000):
    t0 = time.time()
    result = metropolis_spin_glass(params, n_steps=n_steps, n_equilibrate=n_steps // 5)
    return result, time.time() - t0

def run_pimc(params, n_trotter=8, n_steps=5000):
    t0 = time.time()
    result = transverse_field_mc(params, n_trotter=n_trotter, n_steps=n_steps, n_equilibrate=n_steps // 5)
    return result, time.time() - t0

def run_vqe(params, n_layers=2, max_steps=200):
    t0 = time.time()
    result = vqe_ground_state(params, n_layers=n_layers, max_steps=max_steps)
    return result, time.time() - t0

def run_qaoa(params, depth=3, max_steps=200):
    t0 = time.time()
    result = qaoa_ground_state(params, depth=depth, max_steps=max_steps)
    return result, time.time() - t0


# ---------------------------------------------------------------------------
# Print result line (for grep extraction)
# ---------------------------------------------------------------------------


def print_result(result: ExperimentResult) -> None:
    """Print result in grep-extractable format."""
    print(f"RESULT|{result.model}|{result.topology}|N={result.n_agents}|"
          f"T={result.temperature:.3f}|Gamma={result.transverse_field:.3f}|"
          f"method={result.method}|"
          f"tau={result.consensus_time}|speedup={result.consensus_speedup:.2f}|"
          f"vol={result.volatility:.4f}|vol_red={result.volatility_reduction:.4f}|"
          f"advantage={result.quantum_advantage:.6f}|"
          f"time={result.wall_time:.2f}s|{result.description}")
