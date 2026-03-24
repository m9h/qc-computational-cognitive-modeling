import sys
import yaml
import numpy as np
import subprocess
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.hypothesis import MinorityGameHypothesis
from engine.validators import validate_minority_game_invariants, PhysicsViolation
from qcccm.games.minority import MinorityGameParams, run_minority_game

def run_null_test():
    """Execute the Agent's TDD null test before running the experiment."""
    print("Running Null Test (TDD Phase)...")
    result = subprocess.run(
        ["pytest", "workspace/null_test.py", "-v"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Null Test FAILED. The experiment cannot proceed.")
        print(result.stdout)
        sys.exit(1)
    print("✅ Null Test PASSED.\n")

def run_experiment(yaml_path: str):
    """Load hypothesis, run the simulation, validate physics, log results."""
    # 1. Parse and Validate Hypothesis YAML
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        hypothesis = MinorityGameHypothesis(**data)
    except Exception as e:
        print(f"❌ Failed to parse hypothesis YAML: {e}")
        sys.exit(1)

    print(f"==================================================")
    print(f"Hypothesis: {hypothesis.name}")
    print(f"Rationale: {hypothesis.rationale}")
    print(f"==================================================")

    best_vol_reduction = 0.0
    best_config = ""
    results_log = []

    # 2. Execute the Sweep
    for M in hypothesis.sweep.memory_values:
        alpha = (2 ** M) / hypothesis.N_agents
        print(f"Running M={M} (alpha={alpha:.4f})...")

        # --- Classical Baseline ---
        classical_vols = []
        for seed in hypothesis.sweep.seeds:
            params = MinorityGameParams(
                n_agents=hypothesis.N_agents, memory=M, n_strategies=2,
                n_rounds=hypothesis.n_rounds, seed=seed,
            )
            result = run_minority_game(params, quantumness=0.0, beta=hypothesis.sweep.beta)
            validate_minority_game_invariants(result, hypothesis.N_agents)
            classical_vols.append(result.volatility)
        vol_classical = np.mean(classical_vols)

        # --- Quantum Agents ---
        for q in hypothesis.sweep.quantumness_values:
            if q == 0.0:
                continue

            quantum_vols = []
            for seed in hypothesis.sweep.seeds:
                # Engine guarantees EXACT SAME seeds are used
                params = MinorityGameParams(
                    n_agents=hypothesis.N_agents, memory=M, n_strategies=2,
                    n_rounds=hypothesis.n_rounds, seed=seed,
                )
                result = run_minority_game(params, quantumness=q, beta=hypothesis.sweep.beta)
                
                # Active Physics Validation
                try:
                    validate_minority_game_invariants(result, hypothesis.N_agents)
                except PhysicsViolation as e:
                    print(f"\n❌ PHYSICS VIOLATION at M={M}, q={q}: {e}")
                    sys.exit(1)
                
                quantum_vols.append(result.volatility)

            vol_quantum = np.mean(quantum_vols)
            vol_red = (vol_classical - vol_quantum) / vol_classical if vol_classical > 0 else 0.0

            log_line = f"RESULT|M={M}|alpha={alpha:.4f}|q={q:.2f}|beta={hypothesis.sweep.beta}|vol_c={vol_classical:.4f}|vol_q={vol_quantum:.4f}|advantage={vol_red:.4f}"
            results_log.append(log_line)

            if vol_red > best_vol_reduction:
                best_vol_reduction = vol_red
                best_config = log_line

    print("\n--- Execution Complete ---")
    for log in results_log:
        print(log)

    print(f"\n⭐ BEST REDUCTION: {best_config}")

if __name__ == "__main__":
    run_null_test()
    run_experiment(Path(__file__).parent.parent / "workspace/hypothesis.yaml")
