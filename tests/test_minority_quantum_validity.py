"""RED-GREEN tests for minority game quantum effects.

These tests determine whether the QuantumAgent produces genuinely quantum
behavior (interference) or is equivalent to a noisy classical agent.

The key discriminator: if quantum(q) ≈ classical(beta_low) for some beta_low,
then "quantumness" is just noise injection, not interference.

Tests are organized as:
- RED tests: things that SHOULD be true for genuine quantum interference
  but may fail (exposing bugs or false claims)
- GREEN tests: things we know are true (baselines, sanity checks)
"""

import numpy as np
import pytest

from qcccm.games.minority import (
    MinorityGameParams,
    QuantumAgent,
    ClassicalAgent,
    run_minority_game,
    _generate_strategies,
    _history_to_index,
)


# ===================================================================
# GREEN: Baseline sanity — these must always pass
# ===================================================================


class TestBaselines:
    """Known-true properties of the minority game."""

    def test_classical_herding_at_low_alpha(self):
        """At alpha << alpha_c, classical agents herd: volatility > 1."""
        params = MinorityGameParams(n_agents=101, memory=1, n_rounds=500, seed=42)
        result = run_minority_game(params, quantumness=0.0, beta=0.1)
        assert result.volatility > 1.0, (
            f"Expected herding (vol > 1) at alpha={2**1/101:.3f}, got {result.volatility:.3f}"
        )

    def test_classical_coordination_at_high_alpha(self):
        """At alpha >> alpha_c, classical agents coordinate: volatility < 1."""
        params = MinorityGameParams(n_agents=101, memory=6, n_rounds=500, seed=42)
        result = run_minority_game(params, quantumness=0.0, beta=0.1)
        assert result.volatility < 0.5, (
            f"Expected coordination (vol < 0.5) at alpha={2**6/101:.3f}, got {result.volatility:.3f}"
        )

    def test_random_agents_volatility_below_herding(self):
        """At beta=0, agents pick random strategies: volatility should be
        below herding (vol < 1) but above coordination minimum.
        Note: beta=0 isn't truly random actions — each strategy is a fixed
        lookup table, so attendance isn't binomial."""
        params = MinorityGameParams(n_agents=101, memory=3, n_rounds=1000, seed=42)
        result = run_minority_game(params, quantumness=0.0, beta=0.0)
        assert result.volatility < 1.5, (
            f"Random strategy selection: expected vol < 1.5, got {result.volatility:.3f}"
        )


# ===================================================================
# RED: Quantum vs noise — the critical discriminator
# ===================================================================


class TestQuantumVsNoise:
    """Distinguish genuine quantum interference from noise injection.

    If quantum(q=X) ≈ classical(beta=very_low), then quantumness is
    just randomizing strategy selection, not producing interference.
    """

    def test_quantum_vs_low_beta_in_herding_phase(self):
        """In the herding phase, compare quantum agent to noisy classical agent.

        If quantum(q=0.5) gives the same volatility as classical(beta≈0),
        then quantum = noise. If they differ, quantum is doing something
        structurally different.
        """
        params = MinorityGameParams(n_agents=101, memory=2, n_rounds=500, seed=42)

        # Quantum agent
        quantum_vols = []
        for seed in range(10):
            p = params._replace(seed=seed)
            r = run_minority_game(p, quantumness=0.5, beta=0.1)
            quantum_vols.append(r.volatility)
        vol_quantum = np.mean(quantum_vols)

        # Very noisy classical agent (beta → 0 = random)
        noisy_vols = []
        for seed in range(10):
            p = params._replace(seed=seed)
            r = run_minority_game(p, quantumness=0.0, beta=0.001)
            noisy_vols.append(r.volatility)
        vol_noisy = np.mean(noisy_vols)

        # Also: moderately noisy classical
        moderate_vols = []
        for seed in range(10):
            p = params._replace(seed=seed)
            r = run_minority_game(p, quantumness=0.0, beta=0.01)
            moderate_vols.append(r.volatility)
        vol_moderate = np.mean(moderate_vols)

        print(f"\n  QUANTUM VS NOISE (herding phase, alpha={2**2/101:.3f}):")
        print(f"  Classical (beta=0.1):   vol = {np.mean([run_minority_game(params._replace(seed=s), quantumness=0.0, beta=0.1).volatility for s in range(10)]):.4f}")
        print(f"  Classical (beta=0.01):  vol = {vol_moderate:.4f}")
        print(f"  Classical (beta=0.001): vol = {vol_noisy:.4f}")
        print(f"  Quantum (q=0.5):        vol = {vol_quantum:.4f}")

        if abs(vol_quantum - vol_noisy) < 0.05:
            pytest.fail(
                f"Quantum (vol={vol_quantum:.4f}) ≈ noisy classical (vol={vol_noisy:.4f}). "
                f"Quantumness is likely just noise injection, not interference."
            )

    def test_quantum_monotonic_in_q(self):
        """Volatility reduction should increase monotonically with quantumness.

        If it's not monotonic, the effect is noise-dependent, not systematic.
        """
        params = MinorityGameParams(n_agents=101, memory=2, n_rounds=500, seed=42)
        q_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

        vols = []
        for q in q_values:
            seed_vols = []
            for seed in range(5):
                p = params._replace(seed=seed)
                r = run_minority_game(p, quantumness=q, beta=0.1)
                seed_vols.append(r.volatility)
            vols.append(np.mean(seed_vols))

        print(f"\n  MONOTONICITY (alpha={2**2/101:.3f}):")
        for q, v in zip(q_values, vols):
            print(f"    q={q:.1f}: vol={v:.4f}")

        # Check monotonicity from q=0.1 onward (q=0 is the classical baseline)
        for i in range(2, len(vols)):
            if vols[i] > vols[i-1] + 0.05:  # allow 0.05 noise tolerance
                pytest.fail(
                    f"Non-monotonic: vol(q={q_values[i]})={vols[i]:.4f} > "
                    f"vol(q={q_values[i-1]})={vols[i-1]:.4f}. "
                    f"Effect is noise-dependent, not systematic quantum."
                )


# ===================================================================
# RED: Continuity at q→0 — the known bug
# ===================================================================


class TestContinuity:
    """The quantum agent must smoothly approach classical as q→0."""

    def test_strategy_probabilities_continuous(self):
        """The measurement probabilities from QuantumAgent.choose() should
        approach classical Boltzmann probabilities as q→0.

        This tests the density matrix + rotation mechanism directly,
        bypassing the game to isolate the agent's decision function.
        """
        rng = np.random.RandomState(42)
        strategies = _generate_strategies(2, 8, rng)
        agent_rng = np.random.RandomState(0)

        # Give the agent some score history
        scores = np.array([3.0, 1.0])

        # Classical probabilities (Boltzmann at beta=1.0)
        logits = 1.0 * scores
        logits = logits - logits.max()
        classical_probs = np.exp(logits) / np.exp(logits).sum()

        # Quantum agent at various q
        q_values = [0.0, 0.001, 0.01, 0.1, 0.3]
        for q in q_values:
            agent = QuantumAgent(strategies, np.random.RandomState(0), quantumness=q, beta=1.0)
            agent.scores = scores.copy()

            # Extract measurement probabilities by running choose many times
            choices = []
            for _ in range(10000):
                agent_rng_fresh = np.random.RandomState(len(choices))
                agent.rng = agent_rng_fresh
                # We need the internal probabilities, not the game action
                # Reconstruct the measurement_probs
                probs = agent._strategy_probs()
                S = len(probs)
                if q > 0 and S > 1:
                    sqrt_p = np.sqrt(np.clip(probs, 1e-12, None))
                    rho_diag = np.diag(probs)
                    coherence = q * np.outer(sqrt_p, sqrt_p)
                    off_diag_mask = 1.0 - np.eye(S)
                    rho = rho_diag + coherence * off_diag_mask
                    rho = rho / np.trace(rho)
                    phase = q * np.pi / 4
                    rotation = np.eye(S) * np.cos(phase)
                    if S == 2:
                        rotation[0, 1] = np.sin(phase)
                        rotation[1, 0] = -np.sin(phase)
                    rho_rotated = rotation @ rho @ rotation.T
                    measurement_probs = np.clip(np.diag(rho_rotated).real, 1e-12, None)
                    measurement_probs = measurement_probs / measurement_probs.sum()
                else:
                    measurement_probs = probs
                break  # only need one computation

            diff = np.max(np.abs(measurement_probs - classical_probs))
            print(f"  q={q:.3f}: measurement_probs={measurement_probs}, "
                  f"classical={classical_probs}, max_diff={diff:.6f}")

            if q <= 0.01:
                assert diff < 0.05, (
                    f"At q={q}, measurement probs differ from classical by {diff:.4f}. "
                    f"QuantumAgent is not continuous at q→0."
                )


# ===================================================================
# RED: Score dynamics — does quantum change learning, not just selection?
# ===================================================================


class TestScoreDynamics:
    """Check whether quantum agents learn differently or just select differently."""

    def test_score_update_identical(self):
        """QuantumAgent.update() must be identical to ClassicalAgent.update().

        If scores diverge, quantum is changing the learning dynamics
        (through different actions leading to different outcomes), not
        just the selection mechanism. This is expected but should be documented.
        """
        rng = np.random.RandomState(42)
        strategies = _generate_strategies(2, 8, rng)

        classical = ClassicalAgent(strategies.copy(), np.random.RandomState(0))
        quantum = QuantumAgent(strategies.copy(), np.random.RandomState(0), quantumness=0.5, beta=1.0)

        # Same initial scores
        classical.scores = np.array([0.0, 0.0])
        quantum.scores = np.array([0.0, 0.0])

        # Same update (same history, same outcome)
        classical.update(3, winning_side=1)
        quantum.update(3, winning_side=1)

        assert np.allclose(classical.scores, quantum.scores), (
            f"Score update differs: classical={classical.scores}, quantum={quantum.scores}"
        )

    def test_quantum_changes_actions_not_just_scores(self):
        """Over many rounds, quantum agents take different actions than classical,
        which leads to different game dynamics. This is the mechanism by which
        quantum affects volatility — through action diversity, not score computation.
        """
        params = MinorityGameParams(n_agents=51, memory=2, n_rounds=200, seed=42)

        classical = run_minority_game(params, quantumness=0.0, beta=0.1)
        quantum = run_minority_game(params, quantumness=0.5, beta=0.1)

        # Attendance patterns should differ
        corr = np.corrcoef(classical.attendance, quantum.attendance)[0, 1]
        print(f"\n  Attendance correlation (classical vs quantum): {corr:.4f}")

        # They should NOT be perfectly correlated (quantum changes actions)
        assert corr < 0.95, (
            f"Attendance correlation {corr:.4f} too high — quantum isn't changing actions"
        )


# ===================================================================
# RED: Phase transition structure
# ===================================================================


class TestPhaseTransition:
    """The quantum effect should interact with the phase transition at alpha_c."""

    def test_quantum_effect_larger_in_herding_phase(self):
        """Volatility reduction should be larger at low alpha (herding)
        than at high alpha (coordination). If equal, quantum isn't
        interacting with the phase structure."""
        n_seeds = 5

        def mean_volatility(M, q):
            vols = []
            for seed in range(n_seeds):
                params = MinorityGameParams(n_agents=101, memory=M, n_rounds=500, seed=seed)
                r = run_minority_game(params, quantumness=q, beta=0.1)
                vols.append(r.volatility)
            return np.mean(vols)

        # Herding phase (alpha ≈ 0.02)
        vol_classical_herd = mean_volatility(1, 0.0)
        vol_quantum_herd = mean_volatility(1, 0.5)
        reduction_herd = (vol_classical_herd - vol_quantum_herd) / max(vol_classical_herd, 1e-10)

        # Coordination phase (alpha ≈ 0.63)
        vol_classical_coord = mean_volatility(6, 0.0)
        vol_quantum_coord = mean_volatility(6, 0.5)
        reduction_coord = (vol_classical_coord - vol_quantum_coord) / max(vol_classical_coord, 1e-10)

        print(f"\n  PHASE-DEPENDENT EFFECT:")
        print(f"  Herding (M=1):      classical={vol_classical_herd:.4f}, quantum={vol_quantum_herd:.4f}, reduction={reduction_herd:.4f}")
        print(f"  Coordination (M=6): classical={vol_classical_coord:.4f}, quantum={vol_quantum_coord:.4f}, reduction={reduction_coord:.4f}")

        assert reduction_herd > reduction_coord + 0.05, (
            f"Quantum reduction in herding ({reduction_herd:.4f}) not larger than "
            f"coordination ({reduction_coord:.4f}). Effect doesn't interact with phase transition."
        )
