"""Minority Game with classical and quantum agents.

The Minority Game (Challet & Zhang, 1997) is a repeated anti-coordination
game where N agents simultaneously choose one of two options (0 or 1).
The minority side wins. Agents adapt using strategies conditioned on a
shared history of recent winning sides.

Key parameters:
- N: number of agents (odd)
- M: memory length (agents see last M winning sides)
- S: strategies per agent (lookup tables: history → action)

Key observable:
- σ²/N: attendance volatility normalised by N.
  σ²/N < 1 means agents coordinate better than random.
  σ²/N > 1 means herding (worse than random).
  Phase transition at α_c = 2^M / N ≈ 0.34.

The quantum extension adds coherence to strategy selection:
- Classical: hard switch to the best-scoring strategy
- Quantum: density matrix over strategies with off-diagonal coherences,
  enabling interference between strategy paths.

References:
    Challet, D. & Zhang, Y.-C. (1997). Emergence of cooperation and
        organization in an evolutionary game. Physica A, 246, 407-418.
    Challet, D., Marsili, M. & Zhang, Y.-C. (2005). Minority Games:
        Interacting Agents in Financial Markets. Oxford University Press.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Game environment
# ---------------------------------------------------------------------------


class MinorityGameParams(NamedTuple):
    """Parameters for the Minority Game."""

    n_agents: int = 101  # odd number
    memory: int = 3  # M: history bits remembered
    n_strategies: int = 2  # S: strategies per agent
    n_rounds: int = 200
    seed: int = 42


class GameHistory(NamedTuple):
    """Recorded history of a minority game run."""

    attendance: np.ndarray  # (n_rounds,) — number choosing option 1
    winners: np.ndarray  # (n_rounds,) — winning side (0 or 1)
    volatility: float  # σ²/N


def _history_to_index(history: np.ndarray) -> int:
    """Convert binary history array to integer index."""
    return int(np.sum(history * (2 ** np.arange(len(history)))))


def _generate_strategies(
    n_strategies: int, n_histories: int, rng: np.random.RandomState
) -> np.ndarray:
    """Generate random strategy lookup tables.

    Returns:
        (n_strategies, n_histories) array of {0, 1} actions.
    """
    return rng.randint(0, 2, size=(n_strategies, n_histories))


# ---------------------------------------------------------------------------
# Classical agent
# ---------------------------------------------------------------------------


class ClassicalAgent:
    """Agent that plays its best-scoring strategy.

    Each agent has S strategies (lookup tables mapping history patterns
    to actions). After each round, all strategies are scored based on
    whether they would have predicted the winning side. The agent plays
    the strategy with the highest cumulative score.
    """

    def __init__(
        self,
        strategies: np.ndarray,
        rng: np.random.RandomState,
    ):
        self.strategies = strategies  # (S, 2^M)
        self.scores = np.zeros(strategies.shape[0])
        self.rng = rng

    def choose(self, history_idx: int) -> int:
        """Choose an action based on the current history."""
        best = np.flatnonzero(self.scores == self.scores.max())
        strategy_idx = self.rng.choice(best)
        return int(self.strategies[strategy_idx, history_idx])

    def update(self, history_idx: int, winning_side: int) -> None:
        """Update strategy scores based on the outcome."""
        for s in range(len(self.scores)):
            if self.strategies[s, history_idx] == winning_side:
                self.scores[s] += 1
            else:
                self.scores[s] -= 1


# ---------------------------------------------------------------------------
# Quantum agent
# ---------------------------------------------------------------------------


class QuantumAgent:
    """Agent using a density matrix over strategies.

    Instead of hard-switching to the best strategy, the quantum agent
    maintains a density matrix ρ over its S strategies. The diagonal
    encodes strategy preferences (like classical scores), and the
    off-diagonal elements encode coherences that allow interference
    between strategies.

    At each round:
    1. Convert strategy scores to Boltzmann probabilities
    2. Build density matrix with coherences (controlled by quantumness)
    3. Select strategy by sampling from the quantum state
    4. Update scores as in classical agent

    The quantumness parameter q ∈ [0, 1] controls coherence:
    - q = 0: equivalent to classical agent (diagonal ρ)
    - q > 0: interference can break herding patterns
    """

    def __init__(
        self,
        strategies: np.ndarray,
        rng: np.random.RandomState,
        quantumness: float = 0.3,
        beta: float = 1.0,
    ):
        self.strategies = strategies  # (S, 2^M)
        self.scores = np.zeros(strategies.shape[0])
        self.rng = rng
        self.quantumness = quantumness
        self.beta = beta  # inverse temperature for Boltzmann

    def _strategy_probs(self) -> np.ndarray:
        """Convert scores to Boltzmann probabilities."""
        logits = self.beta * self.scores
        logits = logits - logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def choose(self, history_idx: int) -> int:
        """Choose action via quantum strategy selection."""
        probs = self._strategy_probs()
        S = len(probs)

        if self.quantumness > 0 and S > 1:
            # Build density matrix with coherences
            sqrt_p = np.sqrt(np.clip(probs, 1e-12, None))
            rho_diag = np.diag(probs)
            coherence = self.quantumness * np.outer(sqrt_p, sqrt_p)
            off_diag_mask = 1.0 - np.eye(S)
            rho = rho_diag + coherence * off_diag_mask
            rho = rho / np.trace(rho)

            # Measure: probabilities are the diagonal of ρ after "interference"
            # Apply a "Hadamard-like" mixing to let coherences interfere
            # then read off diagonal
            eigvals, eigvecs = np.linalg.eigh(rho)
            eigvals = np.clip(eigvals, 0, None)
            # Slightly rotate the measurement basis (interference effect)
            phase = self.quantumness * np.pi / 4
            rotation = np.eye(S) * np.cos(phase)
            if S == 2:
                rotation[0, 1] = np.sin(phase)
                rotation[1, 0] = -np.sin(phase)
            rho_rotated = rotation @ rho @ rotation.T
            measurement_probs = np.clip(np.diag(rho_rotated).real, 1e-12, None)
            measurement_probs = measurement_probs / measurement_probs.sum()
        else:
            measurement_probs = probs

        strategy_idx = self.rng.choice(S, p=measurement_probs)
        return int(self.strategies[strategy_idx, history_idx])

    def update(self, history_idx: int, winning_side: int) -> None:
        """Update strategy scores (same as classical)."""
        for s in range(len(self.scores)):
            if self.strategies[s, history_idx] == winning_side:
                self.scores[s] += 1
            else:
                self.scores[s] -= 1


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------


def run_minority_game(
    params: MinorityGameParams,
    quantumness: float = 0.0,
    beta: float = 0.1,
) -> GameHistory:
    """Run a complete minority game simulation.

    Args:
        params: game parameters.
        quantumness: 0.0 for classical, > 0 for quantum agents.
        beta: inverse temperature for strategy selection.

    Returns:
        GameHistory with attendance, winners, and volatility.
    """
    N = params.n_agents
    M = params.memory
    S = params.n_strategies
    n_histories = 2**M
    rng = np.random.RandomState(params.seed)

    # Generate agents with random strategies
    agents = []
    for _ in range(N):
        strats = _generate_strategies(S, n_histories, rng)
        agent_rng = np.random.RandomState(rng.randint(0, 2**31))
        if quantumness > 0:
            agents.append(QuantumAgent(strats, agent_rng, quantumness, beta))
        else:
            agents.append(ClassicalAgent(strats, agent_rng))

    # Initialise history with random bits
    history = rng.randint(0, 2, size=M)

    attendance = np.zeros(params.n_rounds)
    winners = np.zeros(params.n_rounds, dtype=int)

    for t in range(params.n_rounds):
        h_idx = _history_to_index(history)

        # All agents choose
        choices = np.array([agent.choose(h_idx) for agent in agents])
        n_ones = int(choices.sum())
        attendance[t] = n_ones

        # Minority wins
        winning_side = 0 if n_ones > N // 2 else 1
        winners[t] = winning_side

        # Update all agents
        for agent in agents:
            agent.update(h_idx, winning_side)

        # Shift history
        history = np.roll(history, -1)
        history[-1] = winning_side

    # Volatility: σ²/N
    sigma2 = np.var(attendance)
    volatility = sigma2 / N

    return GameHistory(
        attendance=attendance,
        winners=winners,
        volatility=volatility,
    )


# ---------------------------------------------------------------------------
# Phase diagram sweep
# ---------------------------------------------------------------------------


def volatility_sweep(
    n_agents: int = 101,
    memory_range: range | None = None,
    n_strategies: int = 2,
    n_rounds: int = 500,
    quantumness: float = 0.0,
    beta: float = 0.1,
    n_seeds: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep memory M to map the phase diagram.

    Returns:
        alpha: (n_points,) — α = 2^M / N
        vol_mean: (n_points,) — mean σ²/N across seeds
        vol_std: (n_points,) — std σ²/N across seeds
    """
    if memory_range is None:
        memory_range = range(1, 8)

    alphas = []
    vol_means = []
    vol_stds = []

    for M in memory_range:
        alpha = 2**M / n_agents
        vols = []
        for seed in range(n_seeds):
            params = MinorityGameParams(
                n_agents=n_agents,
                memory=M,
                n_strategies=n_strategies,
                n_rounds=n_rounds,
                seed=seed,
            )
            result = run_minority_game(params, quantumness=quantumness, beta=beta)
            vols.append(result.volatility)

        alphas.append(alpha)
        vol_means.append(np.mean(vols))
        vol_stds.append(np.std(vols))

    return np.array(alphas), np.array(vol_means), np.array(vol_stds)
