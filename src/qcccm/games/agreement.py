"""Multi-agent agreement as an Ising model.

Formalises the N-agent, M-objective agreement problem (Nayebi, 2025)
as a spin system, enabling direct simulation of alignment complexity
barriers using statistical mechanics.

Each agent holds beliefs (spins) on M binary questions. Agreement
requires all agents to converge within ε. The interaction structure
creates frustration when objectives conflict.

The Ising mapping:
- Spin s_i ∈ {-1, +1}: agent i's belief on a binary question
- Coupling J_ij: interaction between agents (positive = ferromagnetic/align,
  negative = anti-ferromagnetic/disagree)
- External field h_i: agent's private prior bias
- Temperature T: noise / bounded rationality
- Transverse field Γ: quantum coherence strength

Key observables:
- Magnetisation |m|: degree of consensus (1 = full agreement)
- Susceptibility χ: sensitivity to perturbation (peaks at phase transition)
- Communication cost: messages exchanged to reach ε-agreement
- Agreement time: rounds until |m| > 1 - ε

References:
    Nayebi, A. (2025). Intrinsic Barriers and Practical Pathways for
        Human-AI Alignment. arXiv:2502.05934. AAAI 2026.
    Challet, D. & Zhang, Y.-C. (1997). Minority Games.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Ising model core
# ---------------------------------------------------------------------------


class IsingParams(NamedTuple):
    """Parameters for a 2D Ising model."""

    L: int = 20  # lattice side length (L×L spins)
    temperature: float = 2.27  # T (T_c ≈ 2.269 for 2D square lattice)
    coupling: float = 1.0  # J (positive = ferromagnetic)
    field: float = 0.0  # h (external field)
    n_steps: int = 1000  # Monte Carlo steps


class IsingState(NamedTuple):
    """State of an Ising simulation."""

    spins: np.ndarray  # (L, L) array of ±1
    energy: float
    magnetisation: float


def ising_energy(spins: np.ndarray, J: float = 1.0, h: float = 0.0) -> float:
    """Compute Ising Hamiltonian: H = -J Σ_{<ij>} s_i s_j - h Σ_i s_i.

    Uses periodic boundary conditions.
    """
    # Nearest-neighbour interaction (periodic BC via roll)
    interaction = np.sum(
        spins * (np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1))
    )
    field_term = np.sum(spins)
    return -J * interaction - h * field_term


def ising_magnetisation(spins: np.ndarray) -> float:
    """Absolute magnetisation per spin: |m| = |Σ s_i| / N."""
    return abs(np.mean(spins))


# ---------------------------------------------------------------------------
# Metropolis-Hastings Monte Carlo
# ---------------------------------------------------------------------------


def metropolis_step(
    spins: np.ndarray,
    temperature: float,
    J: float = 1.0,
    h: float = 0.0,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """One full sweep of Metropolis-Hastings updates.

    Proposes single spin flips, accepts with Boltzmann probability.
    One sweep = L² attempted flips.

    Args:
        spins: (L, L) spin configuration.
        temperature: T (controls acceptance rate).
        J: coupling constant.
        h: external field.
        rng: random state.

    Returns:
        Updated (L, L) spin configuration.
    """
    if rng is None:
        rng = np.random.RandomState()

    L = spins.shape[0]
    spins = spins.copy()
    beta = 1.0 / max(temperature, 1e-10)

    for _ in range(L * L):
        i = rng.randint(0, L)
        j = rng.randint(0, L)

        # Local energy change for flipping spin (i, j)
        s = spins[i, j]
        neighbours = (
            spins[(i + 1) % L, j]
            + spins[(i - 1) % L, j]
            + spins[i, (j + 1) % L]
            + spins[i, (j - 1) % L]
        )
        delta_E = 2.0 * s * (J * neighbours + h)

        # Metropolis acceptance
        if delta_E <= 0 or rng.random() < np.exp(-beta * delta_E):
            spins[i, j] = -s

    return spins


def run_ising(
    params: IsingParams,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Ising model Monte Carlo simulation.

    Args:
        params: simulation parameters.
        seed: random seed.

    Returns:
        energies: (n_steps,) energy per step.
        magnetisations: (n_steps,) |m| per step.
        final_spins: (L, L) final spin configuration.
    """
    rng = np.random.RandomState(seed)
    spins = rng.choice([-1, 1], size=(params.L, params.L))

    energies = np.zeros(params.n_steps)
    mags = np.zeros(params.n_steps)

    for t in range(params.n_steps):
        spins = metropolis_step(spins, params.temperature, params.coupling, params.field, rng)
        energies[t] = ising_energy(spins, params.coupling, params.field) / (params.L ** 2)
        mags[t] = ising_magnetisation(spins)

    return energies, mags, spins


def phase_diagram(
    L: int = 20,
    temperatures: np.ndarray | None = None,
    n_steps: int = 1000,
    n_equilibrate: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep temperature to map the Ising phase diagram.

    Args:
        L: lattice side length.
        temperatures: array of T values to sweep.
        n_steps: total MC steps per temperature.
        n_equilibrate: steps to discard for equilibration.
        seed: random seed.

    Returns:
        temperatures: (n_T,)
        mean_mag: (n_T,) mean |m| after equilibration.
        mean_energy: (n_T,) mean E/N after equilibration.
        susceptibility: (n_T,) χ = N * Var(m) / T.
    """
    if temperatures is None:
        temperatures = np.linspace(1.0, 4.0, 25)

    mean_mag = np.zeros(len(temperatures))
    mean_energy = np.zeros(len(temperatures))
    susceptibility = np.zeros(len(temperatures))
    N = L * L

    for idx, T in enumerate(temperatures):
        params = IsingParams(L=L, temperature=T, n_steps=n_steps)
        energies, mags, _ = run_ising(params, seed=seed)

        # Discard equilibration
        mags_eq = mags[n_equilibrate:]
        energies_eq = energies[n_equilibrate:]

        mean_mag[idx] = np.mean(mags_eq)
        mean_energy[idx] = np.mean(energies_eq)
        susceptibility[idx] = N * np.var(mags_eq) / max(T, 1e-10)

    return temperatures, mean_mag, mean_energy, susceptibility


# ---------------------------------------------------------------------------
# Multi-agent agreement simulation
# ---------------------------------------------------------------------------


class AgreementParams(NamedTuple):
    """Parameters for a multi-agent agreement simulation."""

    n_agents: int = 20  # N
    n_objectives: int = 5  # M
    epsilon: float = 0.1  # agreement tolerance
    temperature: float = 1.0  # bounded rationality (higher = noisier)
    frustration: float = 0.0  # fraction of anti-ferromagnetic bonds
    max_rounds: int = 500
    seed: int = 42


class AgreementResult(NamedTuple):
    """Result of a multi-agent agreement simulation."""

    agreement_time: int  # rounds to reach ε-agreement (-1 if not reached)
    messages_exchanged: int  # total pairwise messages
    final_disagreement: float  # max pairwise belief distance
    magnetisation_trajectory: np.ndarray  # (rounds,) consensus over time
    method: str


def _generate_couplings(
    n_agents: int,
    frustration: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate agent coupling matrix with optional frustration.

    Args:
        n_agents: N.
        frustration: fraction of bonds that are anti-ferromagnetic.
        rng: random state.

    Returns:
        (N, N) symmetric coupling matrix with ±1 entries.
    """
    J = np.ones((n_agents, n_agents))
    # Randomly flip some bonds to anti-ferromagnetic
    n_bonds = n_agents * (n_agents - 1) // 2
    n_frustrated = int(frustration * n_bonds)
    if n_frustrated > 0:
        bond_idx = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                bond_idx += 1
        bonds = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                bonds.append((i, j))
        frustrated = rng.choice(len(bonds), size=n_frustrated, replace=False)
        for idx in frustrated:
            i, j = bonds[idx]
            J[i, j] = -1
            J[j, i] = -1

    np.fill_diagonal(J, 0)
    return J


def run_agreement_simulation(
    params: AgreementParams,
    quantumness: float = 0.0,
) -> AgreementResult:
    """Simulate multi-agent agreement as spin dynamics.

    Each agent holds beliefs on M binary objectives (spins ∈ {-1, +1}).
    At each round, agents exchange messages with neighbours and update
    beliefs via Metropolis-like dynamics.

    Classical: standard Metropolis acceptance.
    Quantum (q > 0): adds noise that decorrelates agents, simulating
    the effect of quantum fluctuations (transverse field).

    Args:
        params: simulation parameters.
        quantumness: Γ — effective transverse field strength.

    Returns:
        AgreementResult with timing and trajectory.
    """
    N = params.n_agents
    M = params.n_objectives
    rng = np.random.RandomState(params.seed)

    # Initial beliefs: random ±1 for each agent × objective
    beliefs = rng.choice([-1, 1], size=(N, M)).astype(float)

    # Private prior biases (heterogeneous agents)
    priors = rng.randn(N, M) * 0.5

    # Coupling matrix
    J = _generate_couplings(N, params.frustration, rng)

    beta = 1.0 / max(params.temperature, 1e-10)
    messages = 0
    mag_trajectory = []
    agreement_time = -1

    for t in range(params.max_rounds):
        # Compute consensus (magnetisation per objective, then average)
        m_per_obj = np.abs(np.mean(beliefs, axis=0))
        m_avg = np.mean(m_per_obj)
        mag_trajectory.append(m_avg)

        # Check ε-agreement
        if m_avg > 1.0 - params.epsilon:
            agreement_time = t
            break

        # Update each agent
        for i in range(N):
            for obj in range(M):
                # Social pressure: weighted sum of neighbours' beliefs
                social = np.sum(J[i, :] * beliefs[:, obj])
                messages += N - 1  # agent i reads all neighbours

                # Local field: prior + social pressure
                local_field = priors[i, obj] + social / N

                # Energy change for flipping
                delta_E = 2.0 * beliefs[i, obj] * local_field

                # Quantum fluctuation: adds random bit-flip probability
                quantum_flip_prob = quantumness * 0.5 if quantumness > 0 else 0.0

                # Metropolis acceptance + quantum tunneling
                if delta_E <= 0:
                    accept = True
                elif rng.random() < np.exp(-beta * delta_E):
                    accept = True
                elif rng.random() < quantum_flip_prob:
                    accept = True  # quantum tunneling
                else:
                    accept = False

                if accept:
                    beliefs[i, obj] = -beliefs[i, obj]

    # Final disagreement: max pairwise L1 distance
    max_disagree = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.mean(np.abs(beliefs[i] - beliefs[j])) / 2.0
            max_disagree = max(max_disagree, d)

    return AgreementResult(
        agreement_time=agreement_time,
        messages_exchanged=messages,
        final_disagreement=max_disagree,
        magnetisation_trajectory=np.array(mag_trajectory),
        method="quantum" if quantumness > 0 else "classical",
    )


def agreement_scaling(
    agent_counts: list[int] | None = None,
    n_objectives: int = 5,
    epsilon: float = 0.1,
    temperature: float = 1.0,
    frustration: float = 0.0,
    quantumness: float = 0.0,
    max_rounds: int = 500,
    n_seeds: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure how agreement cost scales with agent count N.

    Returns:
        agent_counts: (n_points,)
        mean_messages: (n_points,) mean messages to agreement.
        std_messages: (n_points,) std across seeds.
    """
    if agent_counts is None:
        agent_counts = [5, 10, 15, 20, 30]

    counts = np.array(agent_counts)
    mean_msgs = np.zeros(len(counts))
    std_msgs = np.zeros(len(counts))

    for idx, N in enumerate(counts):
        msgs = []
        for seed in range(n_seeds):
            params = AgreementParams(
                n_agents=N,
                n_objectives=n_objectives,
                epsilon=epsilon,
                temperature=temperature,
                frustration=frustration,
                max_rounds=max_rounds,
                seed=seed,
            )
            result = run_agreement_simulation(params, quantumness=quantumness)
            msgs.append(result.messages_exchanged)
        mean_msgs[idx] = np.mean(msgs)
        std_msgs[idx] = np.std(msgs)

    return counts, mean_msgs, std_msgs
