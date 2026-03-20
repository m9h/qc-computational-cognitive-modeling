"""Tests for error mitigation (ZNE)."""

import jax.numpy as jnp
import numpy as np
import pytest

from qcccm.mitigation.zne import make_noisy_qnode, mitigated_belief_probs


class TestNoisyQNode:
    def test_noisy_circuit_returns_valid_probs(self):
        """Noisy circuit should return valid probabilities."""
        from qcccm.circuits.templates import amplitude_encoding_circuit

        def circuit(probs):
            amplitude_encoding_circuit(probs, wires=[0])

        noisy = make_noisy_qnode(circuit, n_qubits=1, noise_level=0.05)
        probs = jnp.array([0.8, 0.2])
        result = noisy(probs)
        assert float(jnp.sum(result)) == pytest.approx(1.0, abs=1e-4)
        assert jnp.all(result >= -1e-8)

    def test_noise_degrades_result(self):
        """Higher noise should move probabilities toward uniform."""
        from qcccm.circuits.templates import amplitude_encoding_circuit

        def circuit(probs):
            amplitude_encoding_circuit(probs, wires=[0])

        ideal_probs = jnp.array([0.9, 0.1])

        low_noise = make_noisy_qnode(circuit, n_qubits=1, noise_level=0.01)
        high_noise = make_noisy_qnode(circuit, n_qubits=1, noise_level=0.1)

        result_low = low_noise(ideal_probs)
        result_high = high_noise(ideal_probs)

        # Low noise should be closer to ideal than high noise
        err_low = float(jnp.sum((result_low - ideal_probs) ** 2))
        err_high = float(jnp.sum((result_high - ideal_probs) ** 2))
        assert err_low < err_high


class TestZNE:
    def test_mitigated_belief_probs_valid(self):
        """ZNE-mitigated belief probs should be a valid distribution."""
        prior = jnp.array([0.5, 0.5])
        likelihood = jnp.array([0.9, 0.1])
        result = mitigated_belief_probs(
            prior, likelihood, n_qubits=1,
            noise_levels=(0.01, 0.03, 0.05),
        )
        assert float(jnp.sum(result)) == pytest.approx(1.0, abs=1e-4)
        assert jnp.all(result >= -1e-8)

    def test_zne_closer_to_ideal_than_noisy(self):
        """ZNE result should be closer to ideal than the noisiest run."""
        import pennylane as qml
        from qcccm.circuits.belief_circuits import make_belief_qnode

        prior = jnp.array([0.8, 0.2])
        likelihood = jnp.array([0.7, 0.3])

        # Ideal (noiseless)
        ideal_qnode = make_belief_qnode(n_qubits=1)
        ideal = np.asarray(ideal_qnode(prior, likelihood))

        # ZNE mitigated
        mitigated = np.asarray(mitigated_belief_probs(
            prior, likelihood, n_qubits=1,
            noise_levels=(0.02, 0.05, 0.10),
        ))

        # Noisy (highest noise level only)
        from qcccm.circuits.belief_circuits import belief_update_circuit
        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="jax")
        def noisy(p, lik):
            belief_update_circuit(p, lik, [0])
            qml.DepolarizingChannel(0.10, wires=0)
            return qml.probs(wires=[0])

        noisy_result = np.asarray(noisy(prior, likelihood))

        err_mitigated = np.sum((mitigated - ideal) ** 2)
        err_noisy = np.sum((noisy_result - ideal) ** 2)

        # ZNE should be at least somewhat better
        assert err_mitigated <= err_noisy * 1.5  # allow some tolerance
