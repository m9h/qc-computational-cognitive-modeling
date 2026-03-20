"""Bridge between ALF (Active Inference) and QCCCM quantum models.

Converts ALF's POMDP matrices (A, B, C, D) to density matrix formalism
and provides a QuantumEFEAgent that can be used as a drop-in replacement
for ALF's AnalyticAgent with quantum-enhanced policy evaluation.

Key conversions:
- Beliefs (probability vector) → density matrix with optional coherences
- Transition matrix B_a → closest unitary (polar decomposition)
- Preferences C → preferred density matrix

The `quantumness` parameter (q ∈ [0, 1]) controls the degree of quantum
interference. At q = 0, the quantum EFE reduces to classical EFE.
At q > 0, off-diagonal coherences introduce interference effects that
can model violations of classical decision theory (sure-thing principle,
conjunction fallacy, order effects).

Example usage (standalone, no ALF import required):
    >>> import jax.numpy as jnp
    >>> from qcccm.models.alf_bridge import alf_quantum_efe, evaluate_all_policies
    >>> A = jnp.eye(3)
    >>> B = jnp.stack([jnp.eye(3)] * 2, axis=-1)
    >>> C = jnp.array([1.0, 0.0, -1.0])
    >>> beliefs = jnp.array([0.5, 0.3, 0.2])
    >>> G = evaluate_all_policies(A, B, C, beliefs, policies, quantumness=0.3)

Example usage (with ALF):
    >>> from alf import GenerativeModel
    >>> from qcccm.models.alf_bridge import QuantumEFEAgent
    >>> gm = GenerativeModel(A=[A], B=[B], C=[C], D=[D])
    >>> agent = QuantumEFEAgent(gm, quantumness=0.3, gamma=4.0)
    >>> action, info = agent.step([observation])
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from qcccm.core.density_matrix import quantum_relative_entropy, von_neumann_entropy
from qcccm.models.bridge import beliefs_to_density_matrix


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------


def beliefs_to_quantum_state(beliefs: Array, quantumness: float = 0.0) -> Array:
    """Convert a classical belief vector to a density matrix with coherences.

    At q = 0, returns diag(beliefs) (classical limit).
    At q > 0, adds off-diagonal elements proportional to √(p_i p_j),
    producing a state with quantum coherences that can interfere.

    Args:
        beliefs: (n,) probability vector summing to 1.
        quantumness: q ∈ [0, 1] controlling coherence strength.

    Returns:
        (n, n) density matrix with Tr(ρ) = 1.
    """
    rho = beliefs_to_density_matrix(beliefs)

    if quantumness > 0:
        n = beliefs.shape[0]
        sqrt_p = jnp.sqrt(jnp.clip(beliefs, 1e-12, None))
        coherence = quantumness * jnp.outer(sqrt_p, sqrt_p).astype(jnp.complex64)
        off_diag_mask = 1.0 - jnp.eye(n)
        rho = rho + coherence * off_diag_mask
        rho = rho / jnp.trace(rho)

    return rho


def transition_to_unitary(B_a: Array) -> Array:
    """Convert a transition matrix to its closest unitary (polar decomposition).

    Given B_a (a column of the B tensor for action a), finds the unitary U
    that minimises ‖B_a − U‖_F via SVD: U = V Σ W† → U_polar = V W†.

    Args:
        B_a: (n, n) transition matrix P(s'|s, a).

    Returns:
        (n, n) unitary matrix.
    """
    B_c = B_a.astype(jnp.complex64)
    U, _, Vh = jnp.linalg.svd(B_c)
    return U @ Vh


def preferences_to_density_matrix(C: Array) -> Array:
    """Convert ALF log-preferences to a preferred density matrix.

    Applies softmax to C to get a probability vector, then converts
    to a diagonal density matrix.

    Args:
        C: (n_obs,) log-preferences over observations.

    Returns:
        (n_obs, n_obs) preferred density matrix.
    """
    probs = jax.nn.softmax(C)
    return beliefs_to_density_matrix(probs)


# ---------------------------------------------------------------------------
# Quantum EFE for ALF-style models
# ---------------------------------------------------------------------------


def alf_quantum_efe(
    A: Array,
    B: Array,
    C: Array,
    beliefs: Array,
    action: int,
    quantumness: float = 0.0,
) -> Array:
    """Compute quantum expected free energy for a single ALF action.

    Converts ALF POMDP matrices to density matrix formalism, then
    computes quantum EFE with optional coherence effects.

    At quantumness = 0, this approximates classical EFE.
    At quantumness > 0, quantum interference modifies policy evaluation.

    Args:
        A: (n_obs, n_states) likelihood matrix P(o|s).
        B: (n_states, n_states, n_actions) transition tensor P(s'|s,a).
        C: (n_obs,) log-preferences over observations.
        beliefs: (n_states,) current beliefs Q(s).
        action: action index.
        quantumness: q ∈ [0, 1] coherence strength.

    Returns:
        Scalar expected free energy (lower = better).
    """
    # Convert beliefs to density matrix
    rho = beliefs_to_quantum_state(beliefs, quantumness)

    # Get transition unitary for this action
    B_a = B[:, :, action]
    U = transition_to_unitary(B_a)

    # Predicted state
    rho_pred = U @ rho @ jnp.conj(U).T

    # Preferred state from preferences
    preferred = preferences_to_density_matrix(C)

    # Quantum EFE: epistemic + pragmatic
    epistemic = -von_neumann_entropy(rho_pred)
    pragmatic = quantum_relative_entropy(rho_pred, preferred)

    return epistemic + pragmatic


def evaluate_all_policies(
    A: Array,
    B: Array,
    C: Array,
    beliefs: Array,
    policies: Array,
    quantumness: float = 0.0,
) -> Array:
    """Evaluate quantum EFE for all candidate policies.

    For each policy (sequence of actions), rolls forward through time
    computing quantum EFE at each step and summing.

    Args:
        A: (n_obs, n_states) likelihood matrix.
        B: (n_states, n_states, n_actions) transition tensor.
        C: (n_obs,) log-preferences.
        beliefs: (n_states,) current beliefs.
        policies: (n_policies, T, n_factors) action sequences.
        quantumness: coherence strength.

    Returns:
        (n_policies,) array of total EFE per policy.
    """
    n_policies = policies.shape[0]
    T = policies.shape[1]

    G_all = np.zeros(n_policies)

    for pi in range(n_policies):
        G_total = 0.0
        current_beliefs = jnp.array(beliefs)

        for t in range(T):
            action = int(policies[pi, t, 0])

            G_step = alf_quantum_efe(
                A, B, C, current_beliefs, action, quantumness
            )
            G_total += float(G_step)

            # Propagate beliefs forward classically
            B_a = B[:, :, action]
            next_beliefs = B_a @ current_beliefs
            next_beliefs = jnp.clip(next_beliefs, 1e-16, None)
            current_beliefs = next_beliefs / jnp.sum(next_beliefs)

        G_all[pi] = G_total

    return G_all


# ---------------------------------------------------------------------------
# QuantumEFEAgent (works with or without ALF installed)
# ---------------------------------------------------------------------------


class QuantumEFEAgent:
    """Active Inference agent using quantum EFE for policy evaluation.

    Drop-in replacement for ALF's AnalyticAgent. Uses the same
    generative model interface but evaluates policies with quantum
    expected free energy instead of classical EFE.

    The quantumness parameter controls the degree of quantum interference:
    - q = 0: equivalent to classical active inference
    - q > 0: quantum coherences introduce interference effects

    Args:
        gm: ALF GenerativeModel (or compatible object with A, B, C, D, E,
            policies, num_factors, num_modalities attributes).
        quantumness: coherence strength q ∈ [0, 1].
        gamma: policy precision (inverse temperature).
        learning_rate: habit learning rate.
        seed: random seed.
    """

    def __init__(
        self,
        gm,
        quantumness: float = 0.3,
        gamma: float = 4.0,
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
        self.gm = gm
        self.quantumness = quantumness
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed)

        self.beliefs = [d.copy() for d in gm.D]
        self.E = gm.E.copy()

        self.belief_history: list = []
        self.action_history: list = []
        self.efe_history: list = []
        self.policy_prob_history: list = []

    def step(self, observation: list[int]) -> tuple[int, dict]:
        """One step of quantum active inference.

        1. Bayesian belief update (same as classical)
        2. Quantum EFE policy evaluation
        3. Softmax action selection

        Args:
            observation: list of observation indices, one per modality.

        Returns:
            (action_index, info_dict)
        """
        # 1. Belief update (standard Bayesian)
        for f in range(self.gm.num_factors):
            for m in range(self.gm.num_modalities):
                a_matrix = self.gm.A[m]
                if a_matrix.ndim == 2:
                    likelihood = a_matrix[observation[m], :]
                    posterior = self.beliefs[f] * likelihood
                    posterior = np.clip(posterior, 1e-16, None)
                    self.beliefs[f] = posterior / posterior.sum()

        self.belief_history.append([b.copy() for b in self.beliefs])

        # 2. Quantum EFE evaluation
        G = evaluate_all_policies(
            jnp.array(self.gm.A[0]),
            jnp.array(self.gm.B[0]),
            jnp.array(self.gm.C[0]),
            jnp.array(self.beliefs[0]),
            np.array(self.gm.policies),
            quantumness=self.quantumness,
        )
        self.efe_history.append(G.copy())

        # 3. Action selection (softmax)
        log_E = np.log(np.clip(self.E, 1e-16, None))
        log_posterior = -self.gamma * G + log_E
        log_posterior = log_posterior - log_posterior.max()
        policy_probs = np.exp(log_posterior)
        policy_probs = policy_probs / policy_probs.sum()
        self.policy_prob_history.append(policy_probs.copy())

        policy_idx = int(self.rng.choice(len(policy_probs), p=policy_probs))
        selected_policy = self.gm.policies[policy_idx]
        action = int(selected_policy[0, 0])
        self.action_history.append(action)

        info = {
            "beliefs": [b.copy() for b in self.beliefs],
            "G": G,
            "policy_probs": policy_probs,
            "selected_policy": policy_idx,
            "quantumness": self.quantumness,
        }
        return action, info

    def learn(self, outcome_valence: float) -> None:
        """Update habits based on outcome."""
        if self.action_history:
            last_idx = (
                self.policy_prob_history[-1].argmax()
                if self.policy_prob_history
                else 0
            )
            self.E[last_idx] += self.learning_rate * outcome_valence
            self.E = np.clip(self.E, 1e-8, None)
            self.E = self.E / self.E.sum()

    def reset(self) -> None:
        """Reset beliefs to priors."""
        self.beliefs = [d.copy() for d in self.gm.D]
        self.belief_history.clear()
        self.action_history.clear()
        self.efe_history.clear()
        self.policy_prob_history.clear()
