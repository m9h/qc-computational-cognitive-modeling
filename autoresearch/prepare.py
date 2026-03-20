"""Autoresearch infrastructure — READ ONLY.

Provides solver wrappers, metric computation, and result logging.
The AI agent should NOT modify this file.
"""

from __future__ import annotations

import csv
import os
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_FILE = Path(__file__).parent / "results.tsv"
LOG_FILE = Path(__file__).parent / "run.log"
TIMEOUT_SECONDS = 600  # 10-minute hard timeout
BRUTE_FORCE_MAX_N = 20  # Exact ground state feasible up to this size


# ---------------------------------------------------------------------------
# Exact ground state (brute force)
# ---------------------------------------------------------------------------


def exact_ground_state(params: SocialSpinGlassParams) -> tuple[float, np.ndarray]:
    """Find exact ground state by exhaustive enumeration.

    Only feasible for N <= BRUTE_FORCE_MAX_N.
    Returns (energy, spin_configuration).
    """
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
# Experiment result
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Container for a single experiment's results."""

    # Identification
    commit: str = ""
    description: str = ""

    # Model specification
    model: str = "SK"  # SK, EA_bimodal, EA_gaussian, custom
    topology: str = "complete"
    disorder: str = "gaussian"
    n_agents: int = 8
    temperature: float = 0.5
    transverse_field: float = 0.0
    frustration: float = 0.0  # for EA bimodal
    seed: int = 42

    # Results
    method: str = "metropolis"  # metropolis, pimc, vqe, qaoa
    E_best: float = 0.0
    E_exact: float = 0.0
    quantum_advantage: float = 0.0
    q_EA: float = 0.0
    wall_time: float = 0.0
    status: str = "ok"  # ok, crash, timeout

    # Secondary metrics
    magnetization: float = 0.0
    frustration_index: float = 0.0
    binder: float = 0.0

    # Additional metadata
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_quantum_advantage(
    E_classical: float,
    E_quantum: float,
    E_exact: float,
) -> float:
    """Compute quantum advantage metric.

    quantum_advantage = (E_classical - E_quantum) / |E_exact|

    Positive means quantum found lower energy. Negative means classical was better.
    """
    if abs(E_exact) < 1e-10:
        return 0.0
    return (E_classical - E_quantum) / abs(E_exact)


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------


HEADER = [
    "commit", "model", "topology", "disorder", "N", "T", "Gamma",
    "frustration", "method", "E_best", "E_exact", "quantum_advantage",
    "q_EA", "magnetization", "frustration_index", "binder",
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
            result.commit,
            result.model,
            result.topology,
            result.disorder,
            result.n_agents,
            f"{result.temperature:.4f}",
            f"{result.transverse_field:.4f}",
            f"{result.frustration:.4f}",
            result.method,
            f"{result.E_best:.6f}",
            f"{result.E_exact:.6f}",
            f"{result.quantum_advantage:.6f}",
            f"{result.q_EA:.6f}",
            f"{result.magnetization:.6f}",
            f"{result.frustration_index:.6f}",
            f"{result.binder:.6f}",
            f"{result.wall_time:.2f}",
            result.status,
            result.description,
        ])


def get_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Solver runners with timing
# ---------------------------------------------------------------------------


def run_classical(params: SocialSpinGlassParams, n_steps: int = 5000) -> tuple[SolverResult, float]:
    """Run Metropolis and return (result, wall_time)."""
    t0 = time.time()
    result = metropolis_spin_glass(params, n_steps=n_steps, n_equilibrate=n_steps // 5)
    return result, time.time() - t0


def run_pimc(
    params: SocialSpinGlassParams,
    n_trotter: int = 8,
    n_steps: int = 5000,
) -> tuple[SolverResult, float]:
    """Run transverse-field PIMC and return (result, wall_time)."""
    t0 = time.time()
    result = transverse_field_mc(
        params, n_trotter=n_trotter, n_steps=n_steps, n_equilibrate=n_steps // 5,
    )
    return result, time.time() - t0


def run_vqe(
    params: SocialSpinGlassParams,
    n_layers: int = 2,
    max_steps: int = 200,
) -> tuple[SolverResult, float]:
    """Run VQE and return (result, wall_time)."""
    t0 = time.time()
    result = vqe_ground_state(params, n_layers=n_layers, max_steps=max_steps)
    return result, time.time() - t0


def run_qaoa(
    params: SocialSpinGlassParams,
    depth: int = 3,
    max_steps: int = 200,
) -> tuple[SolverResult, float]:
    """Run QAOA and return (result, wall_time)."""
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
          f"method={result.method}|E={result.E_best:.6f}|E_exact={result.E_exact:.6f}|"
          f"advantage={result.quantum_advantage:.6f}|"
          f"q_EA={result.q_EA:.4f}|time={result.wall_time:.2f}s|"
          f"{result.description}")
