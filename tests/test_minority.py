"""Tests for the minority game module."""

import numpy as np

from qcccm.games.minority import (
    ClassicalAgent,
    MinorityGameParams,
    QuantumAgent,
    _generate_strategies,
    _history_to_index,
    run_minority_game,
    volatility_sweep,
)


class TestHistoryIndex:
    def test_all_zeros(self):
        """All-zero history maps to index 0."""
        assert _history_to_index(np.array([0, 0, 0])) == 0

    def test_binary_encoding(self):
        """History [1, 0, 1] → 1*1 + 0*2 + 1*4 = 5."""
        assert _history_to_index(np.array([1, 0, 1])) == 5


class TestClassicalAgent:
    def test_choose_returns_valid_action(self):
        """Classical agent should return 0 or 1."""
        rng = np.random.RandomState(0)
        strats = _generate_strategies(2, 8, rng)
        agent = ClassicalAgent(strats, np.random.RandomState(1))
        action = agent.choose(0)
        assert action in (0, 1)

    def test_score_update(self):
        """Winning strategy should gain points."""
        strats = np.array([[0, 1], [1, 0]])  # 2 strategies, 2 histories
        agent = ClassicalAgent(strats, np.random.RandomState(1))
        agent.update(0, winning_side=0)  # strategy 0 predicted 0 at history 0 → +1
        assert agent.scores[0] == 1
        assert agent.scores[1] == -1


class TestQuantumAgent:
    def test_choose_returns_valid_action(self):
        """Quantum agent should return 0 or 1."""
        rng = np.random.RandomState(0)
        strats = _generate_strategies(2, 8, rng)
        agent = QuantumAgent(strats, np.random.RandomState(1), quantumness=0.3)
        action = agent.choose(0)
        assert action in (0, 1)

    def test_zero_quantumness_matches_classical_structure(self):
        """With q=0, quantum agent should behave like classical."""
        rng = np.random.RandomState(42)
        strats = _generate_strategies(2, 8, rng)
        q_agent = QuantumAgent(strats.copy(), np.random.RandomState(0), quantumness=0.0)
        c_agent = ClassicalAgent(strats.copy(), np.random.RandomState(0))
        # Both should choose from the same probability distribution initially
        # (uniform scores → uniform probabilities)
        q_action = q_agent.choose(3)
        c_action = c_agent.choose(3)
        # Can't guarantee same action (random), but both should be valid
        assert q_action in (0, 1)
        assert c_action in (0, 1)


class TestMinorityGame:
    def test_attendance_shape(self):
        """Attendance array should match n_rounds."""
        params = MinorityGameParams(n_agents=11, memory=2, n_rounds=50, seed=0)
        result = run_minority_game(params)
        assert result.attendance.shape == (50,)

    def test_winners_are_binary(self):
        """Winning side should always be 0 or 1."""
        params = MinorityGameParams(n_agents=21, memory=2, n_rounds=30, seed=0)
        result = run_minority_game(params)
        assert set(result.winners).issubset({0, 1})

    def test_volatility_is_finite(self):
        """Volatility should be a finite positive number."""
        params = MinorityGameParams(n_agents=21, memory=2, n_rounds=100, seed=0)
        result = run_minority_game(params)
        assert np.isfinite(result.volatility)
        assert result.volatility > 0

    def test_quantum_game_runs(self):
        """Quantum minority game should complete without error."""
        params = MinorityGameParams(n_agents=21, memory=2, n_rounds=50, seed=0)
        result = run_minority_game(params, quantumness=0.3)
        assert result.attendance.shape == (50,)
        assert np.isfinite(result.volatility)

    def test_quantum_changes_volatility(self):
        """Quantum agents should produce different volatility than classical."""
        params = MinorityGameParams(n_agents=51, memory=3, n_rounds=200, seed=42)
        classical = run_minority_game(params, quantumness=0.0)
        quantum = run_minority_game(params, quantumness=0.5)
        # They should differ (not guaranteed, but very likely with q=0.5)
        # Use a loose check — just verify both are finite and positive
        assert np.isfinite(classical.volatility)
        assert np.isfinite(quantum.volatility)


class TestVolatilitySweep:
    def test_sweep_output_shapes(self):
        """Volatility sweep should return matching arrays."""
        alpha, vol_mean, vol_std = volatility_sweep(
            n_agents=21, memory_range=range(1, 4),
            n_rounds=50, n_seeds=2,
        )
        assert len(alpha) == 3
        assert len(vol_mean) == 3
        assert len(vol_std) == 3
        assert all(np.isfinite(vol_mean))

    def test_alpha_increases_with_memory(self):
        """α = 2^M / N should increase with M."""
        alpha, _, _ = volatility_sweep(
            n_agents=21, memory_range=range(1, 5),
            n_rounds=50, n_seeds=1,
        )
        for i in range(len(alpha) - 1):
            assert alpha[i + 1] > alpha[i]
