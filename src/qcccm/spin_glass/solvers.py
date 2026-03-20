"""Classical and quantum solvers for social spin glass models.

Provides:
- Metropolis Monte Carlo (classical baseline)
- Transverse-field Monte Carlo (quantum-inspired classical)
- VQE via PennyLane (variational quantum eigensolver)
- QAOA via PennyLane (quantum approximate optimization)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from qcccm.spin_glass.hamiltonians import SocialSpinGlassParams, social_hamiltonian_classical


class SolverResult(NamedTuple):
    """Result from a spin glass solver.

    Attributes:
        spins: (N,) best spin configuration found.
        energy: Energy of best configuration.
        trajectory: (n_steps, N) spin trajectory (for MC methods).
        energies: (n_steps,) energy trajectory.
        method: Name of the solver used.
        metadata: Additional solver-specific info.
    """

    spins: np.ndarray
    energy: float
    trajectory: np.ndarray | None = None
    energies: np.ndarray | None = None
    method: str = "unknown"
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Classical Metropolis
# ---------------------------------------------------------------------------


def metropolis_spin_glass(
    params: SocialSpinGlassParams,
    n_steps: int = 5000,
    n_equilibrate: int = 1000,
) -> SolverResult:
    """Standard Metropolis Monte Carlo for a spin glass.

    Single-spin-flip dynamics at fixed temperature.

    Args:
        params: Spin glass parameters (includes J, fields, temperature).
        n_steps: Total MC sweeps.
        n_equilibrate: Sweeps to discard before recording trajectory.

    Returns:
        SolverResult with best configuration, energy trajectory, and spin trajectory.
    """
    rng = np.random.default_rng(params.seed)
    N = params.n_agents
    T = params.temperature

    spins = rng.choice([-1, 1], size=N).astype(float)
    energy = social_hamiltonian_classical(params, spins)

    best_spins = spins.copy()
    best_energy = energy

    trajectory = []
    energies = []

    for step in range(n_steps):
        # Single spin flip
        i = rng.integers(0, N)
        delta_E = 2.0 * spins[i] * (
            np.dot(params.J[i], spins) + params.fields[i]
        )

        if delta_E <= 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
            spins[i] *= -1
            energy += delta_E

        if energy < best_energy:
            best_energy = energy
            best_spins = spins.copy()

        if step >= n_equilibrate:
            trajectory.append(spins.copy())
            energies.append(energy)

    return SolverResult(
        spins=best_spins,
        energy=best_energy,
        trajectory=np.array(trajectory) if trajectory else None,
        energies=np.array(energies) if energies else None,
        method="metropolis",
    )


# ---------------------------------------------------------------------------
# Transverse-field Monte Carlo (quantum-inspired)
# ---------------------------------------------------------------------------


def transverse_field_mc(
    params: SocialSpinGlassParams,
    n_trotter: int = 8,
    n_steps: int = 5000,
    n_equilibrate: int = 1000,
) -> SolverResult:
    """Path-integral Monte Carlo for transverse-field spin glass.

    Simulates the quantum partition function by introducing Trotter replicas
    along an imaginary-time direction. Each spin s_i becomes a chain of P
    replicas coupled ferromagnetically. This is the standard method for
    simulated quantum annealing (SQA).

    The transverse field Gamma from params introduces quantum tunneling:
    J_tau = -(T/2) ln(tanh(Gamma / (P*T))) between Trotter slices.

    Args:
        params: Spin glass parameters. transverse_field > 0 enables tunneling.
        n_trotter: Number of Trotter slices (imaginary time discretization).
        n_steps: Total MC sweeps (each sweep = N*P single-spin flips).
        n_equilibrate: Sweeps to discard.

    Returns:
        SolverResult with best configuration (from the first Trotter slice).
    """
    rng = np.random.default_rng(params.seed)
    N = params.n_agents
    P = n_trotter
    T = max(params.temperature, 1e-10)
    Gamma = params.transverse_field

    # Trotter coupling: ferromagnetic along imaginary time
    if Gamma > 0:
        J_tau = -0.5 * T * np.log(np.tanh(Gamma / (P * T) + 1e-15))
    else:
        J_tau = 0.0

    # Initialize P copies of N spins
    replicas = rng.choice([-1, 1], size=(P, N)).astype(float)
    T_eff = T * P  # effective temperature for in-slice interactions

    best_spins = replicas[0].copy()
    best_energy = social_hamiltonian_classical(params, best_spins)

    trajectory = []
    energies = []

    for step in range(n_steps):
        # Sweep: try flipping each spin in each replica
        for _ in range(N * P):
            p = rng.integers(0, P)
            i = rng.integers(0, N)

            # In-slice energy change (social interactions at effective T)
            delta_E_in = 2.0 * replicas[p, i] * (
                np.dot(params.J[i], replicas[p]) / P + params.fields[i] / P
            )

            # Inter-slice energy change (Trotter coupling)
            p_prev = (p - 1) % P
            p_next = (p + 1) % P
            delta_E_tau = 2.0 * J_tau * replicas[p, i] * (
                replicas[p_prev, i] + replicas[p_next, i]
            )

            delta_E = delta_E_in + delta_E_tau

            if delta_E <= 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
                replicas[p, i] *= -1

        # Track the physical (first) replica
        phys_spins = replicas[0]
        phys_energy = social_hamiltonian_classical(params, phys_spins)

        if phys_energy < best_energy:
            best_energy = phys_energy
            best_spins = phys_spins.copy()

        if step >= n_equilibrate:
            trajectory.append(phys_spins.copy())
            energies.append(phys_energy)

    return SolverResult(
        spins=best_spins,
        energy=best_energy,
        trajectory=np.array(trajectory) if trajectory else None,
        energies=np.array(energies) if energies else None,
        method=f"transverse_field_mc(P={P})",
        metadata={"n_trotter": P, "J_tau": float(J_tau)},
    )


# ---------------------------------------------------------------------------
# VQE via PennyLane
# ---------------------------------------------------------------------------


def vqe_ground_state(
    params: SocialSpinGlassParams,
    n_layers: int = 2,
    max_steps: int = 200,
    learning_rate: float = 0.1,
) -> SolverResult:
    """Find the ground state of the social Hamiltonian via VQE.

    Uses a hardware-efficient ansatz (RY-RZ rotations + CNOT ladder)
    optimized with Adam via JAX autodiff.

    Args:
        params: Spin glass parameters.
        n_layers: Depth of the variational ansatz.
        max_steps: Optimization steps.
        learning_rate: Adam learning rate.

    Returns:
        SolverResult with best spin configuration from measurement.
    """
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    N = params.n_agents
    H = _import_hamiltonian(params)

    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="jax")
    def cost_fn(theta):
        for layer in range(n_layers):
            for i in range(N):
                qml.RY(theta[layer, i, 0], wires=i)
                qml.RZ(theta[layer, i, 1], wires=i)
            for i in range(N - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(H)

    @qml.qnode(dev, interface="jax")
    def measure_fn(theta):
        for layer in range(n_layers):
            for i in range(N):
                qml.RY(theta[layer, i, 0], wires=i)
                qml.RZ(theta[layer, i, 1], wires=i)
            for i in range(N - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]

    # Initialize parameters
    key = jax.random.PRNGKey(params.seed)
    theta = jax.random.normal(key, shape=(n_layers, N, 2)) * 0.1

    # Optimize with simple gradient descent (no optax dependency required)
    energy_trajectory = []
    best_energy = float("inf")
    best_theta = theta

    grad_fn = jax.jit(jax.grad(cost_fn))
    cost_fn_jit = jax.jit(cost_fn)

    for step in range(max_steps):
        grads = grad_fn(theta)
        theta = theta - learning_rate * grads
        energy = float(cost_fn_jit(theta))
        energy_trajectory.append(energy)

        if energy < best_energy:
            best_energy = energy
            best_theta = theta

    # Extract spin configuration from best parameters
    z_expectations = np.array(measure_fn(best_theta))
    spins = np.sign(z_expectations)
    spins[spins == 0] = 1.0  # break ties

    return SolverResult(
        spins=spins,
        energy=best_energy,
        energies=np.array(energy_trajectory),
        method=f"vqe(layers={n_layers})",
        metadata={"n_layers": n_layers, "final_params": np.array(best_theta)},
    )


# ---------------------------------------------------------------------------
# QAOA via PennyLane
# ---------------------------------------------------------------------------


def qaoa_ground_state(
    params: SocialSpinGlassParams,
    depth: int = 3,
    max_steps: int = 200,
    learning_rate: float = 0.1,
) -> SolverResult:
    """Find the ground state via QAOA (Quantum Approximate Optimization Algorithm).

    The cost Hamiltonian encodes the social spin glass energy.
    The mixer is the standard X-mixer (transverse field).

    Args:
        params: Spin glass parameters.
        depth: QAOA depth p (number of cost+mixer layers).
        max_steps: Optimization steps.
        learning_rate: Gradient descent step size.

    Returns:
        SolverResult with best spin configuration.
    """
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    N = params.n_agents
    H_cost = _import_hamiltonian(params)

    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="jax")
    def qaoa_cost(gammas, betas):
        # Initial superposition
        for i in range(N):
            qml.Hadamard(wires=i)

        for layer in range(depth):
            # Cost layer: exp(-i gamma H_cost)
            qml.ApproxTimeEvolution(H_cost, gammas[layer], 1)
            # Mixer layer: exp(-i beta X)
            for i in range(N):
                qml.RX(2 * betas[layer], wires=i)

        return qml.expval(H_cost)

    @qml.qnode(dev, interface="jax")
    def qaoa_measure(gammas, betas):
        for i in range(N):
            qml.Hadamard(wires=i)
        for layer in range(depth):
            qml.ApproxTimeEvolution(H_cost, gammas[layer], 1)
            for i in range(N):
                qml.RX(2 * betas[layer], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]

    key = jax.random.PRNGKey(params.seed)
    k1, k2 = jax.random.split(key)
    gammas = jax.random.uniform(k1, shape=(depth,)) * 0.5
    betas = jax.random.uniform(k2, shape=(depth,)) * 0.5

    energy_trajectory = []
    best_energy = float("inf")
    best_gammas, best_betas = gammas, betas

    grad_fn = jax.jit(jax.grad(qaoa_cost, argnums=(0, 1)))
    cost_jit = jax.jit(qaoa_cost)

    for step in range(max_steps):
        g_gamma, g_beta = grad_fn(gammas, betas)
        gammas = gammas - learning_rate * g_gamma
        betas = betas - learning_rate * g_beta
        energy = float(cost_jit(gammas, betas))
        energy_trajectory.append(energy)

        if energy < best_energy:
            best_energy = energy
            best_gammas, best_betas = gammas, betas

    z_expectations = np.array(qaoa_measure(best_gammas, best_betas))
    spins = np.sign(z_expectations)
    spins[spins == 0] = 1.0

    return SolverResult(
        spins=spins,
        energy=best_energy,
        energies=np.array(energy_trajectory),
        method=f"qaoa(p={depth})",
        metadata={"depth": depth, "gammas": np.array(best_gammas), "betas": np.array(best_betas)},
    )


def _import_hamiltonian(params: SocialSpinGlassParams):
    """Build PennyLane Hamiltonian, importing lazily."""
    from qcccm.spin_glass.hamiltonians import social_hamiltonian_pennylane

    return social_hamiltonian_pennylane(params)
