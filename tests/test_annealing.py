"""Tests for the annealing module (no dwave required)."""

import numpy as np

from qcccm.annealing.qubo import (
    decode_qubo_solution,
    efe_to_qubo,
)
from qcccm.annealing.solve import (
    _evaluate_qubo,
    solve_policy_assignment,
    solve_policy_qubo,
)


class TestQUBO:
    def test_qubo_variable_count(self):
        """For N agents, K policies: N*K binary variables."""
        efe = np.array([[1.0, 2.0], [3.0, 0.5], [1.5, 1.0]])
        adj = np.ones((3, 3)) - np.eye(3)
        Q = efe_to_qubo(efe, adj)
        # All variable indices should be in [0, 6)
        all_vars = set()
        for i, j in Q.keys():
            all_vars.add(i)
            all_vars.add(j)
        assert max(all_vars) < 6  # 3 agents * 2 policies

    def test_one_hot_violation_increases_energy(self):
        """Violating one-hot (two policies active) should cost more."""
        efe = np.array([[0.0, 0.0], [0.0, 0.0]])
        adj = np.zeros((2, 2))
        Q = efe_to_qubo(efe, adj, penalty_strength=10.0)

        # Valid: agent 0 → policy 0, agent 1 → policy 1
        valid = {0: 1, 1: 0, 2: 0, 3: 1}
        e_valid = _evaluate_qubo(Q, valid)

        # Invalid: agent 0 has both policies active
        invalid = {0: 1, 1: 1, 2: 0, 3: 1}
        e_invalid = _evaluate_qubo(Q, invalid)

        assert e_invalid > e_valid

    def test_decode_roundtrip(self):
        """decode should recover the known assignment."""
        sample = {0: 0, 1: 1, 2: 1, 3: 0}  # agent 0→policy 1, agent 1→policy 0
        result = decode_qubo_solution(sample, n_agents=2, n_policies=2)
        assert result[0] == 1
        assert result[1] == 0

    def test_efe_to_qubo_keys_valid(self):
        """QUBO keys should be (i, j) with i <= j."""
        efe = np.array([[1.0, 2.0], [3.0, 0.5]])
        adj = np.array([[0, 1], [1, 0]])
        Q = efe_to_qubo(efe, adj)
        for i, j in Q.keys():
            assert i <= j


class TestSolver:
    def test_brute_force_small_problem(self):
        """Brute force on a 2-agent, 2-policy problem."""
        efe = np.array([[1.0, 5.0], [5.0, 1.0]])
        adj = np.zeros((2, 2))
        Q = efe_to_qubo(efe, adj, interaction_strength=0.0)
        sol = solve_policy_qubo(Q, n_agents=2, n_policies=2, method="brute_force")
        # Agent 0 should pick policy 0 (EFE=1), agent 1 should pick policy 1 (EFE=1)
        assert sol.assignments[0] == 0
        assert sol.assignments[1] == 1

    def test_solve_policy_assignment_end_to_end(self):
        """End-to-end solve with brute force."""
        efe = np.array([[0.5, 3.0], [3.0, 0.5], [1.0, 2.0]])
        adj = np.ones((3, 3)) - np.eye(3)
        sol = solve_policy_assignment(
            efe, adj, method="brute_force",
            interaction_strength=0.1, penalty_strength=10.0,
        )
        assert sol.assignments.shape == (3,)
        assert all(0 <= a < 2 for a in sol.assignments)
        assert np.isfinite(sol.energy)

    def test_simulated_solver_runs(self):
        """Simulated/random solver should run without dwave."""
        efe = np.array([[1.0, 2.0], [2.0, 1.0]])
        adj = np.array([[0, 1], [1, 0]])
        Q = efe_to_qubo(efe, adj)
        sol = solve_policy_qubo(Q, n_agents=2, n_policies=2, method="simulated", num_reads=50)
        assert sol.assignments.shape == (2,)
        assert np.isfinite(sol.energy)
