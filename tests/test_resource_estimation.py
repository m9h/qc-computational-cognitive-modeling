"""Tests for qubit resource estimation."""

import math

from qcccm.neuroai.resource_estimation import (
    NEURAL_SYSTEMS,
    NEUROTRANSMITTERS,
    estimate_molecular_qubits,
    estimate_neural_qubits,
    full_resource_table,
    parse_molecular_formula,
)


class TestNeuralQubitEstimation:
    def test_minicolumn(self):
        """90 neurons → ⌈log₂ 90⌉ = 7 amplitude qubits."""
        est = estimate_neural_qubits(NEURAL_SYSTEMS["cortical_minicolumn"])
        assert est.amplitude_qubits == 7
        assert est.basis_state_qubits == 90

    def test_c_elegans(self):
        """302 neurons → ⌈log₂ 302⌉ = 9 amplitude qubits."""
        est = estimate_neural_qubits(NEURAL_SYSTEMS["c_elegans"])
        assert est.amplitude_qubits == 9

    def test_human(self):
        """86 billion neurons → ⌈log₂ 86e9⌉ = 37 amplitude qubits."""
        est = estimate_neural_qubits(NEURAL_SYSTEMS["human"])
        assert est.amplitude_qubits == math.ceil(math.log2(86_000_000_000))
        assert est.amplitude_qubits == 37

    def test_organoid(self):
        """Organoid systems should have reasonable qubit counts."""
        est = estimate_neural_qubits(NEURAL_SYSTEMS["organoid_mature"])
        assert est.amplitude_qubits == 17  # ⌈log₂ 100000⌉
        assert est.basis_state_qubits == 100_000


class TestMolecularFormulaParsing:
    def test_gaba(self):
        """C4H9NO2 → C:4, H:9, N:1, O:2."""
        comp = parse_molecular_formula("C4H9NO2")
        assert comp["C"] == 4
        assert comp["H"] == 9
        assert comp["N"] == 1
        assert comp["O"] == 2

    def test_glycine(self):
        """NH2CH2COOH → N:1, H:4, C:2, O:2."""
        comp = parse_molecular_formula("NH2CH2COOH")
        assert comp["N"] == 1
        assert comp["C"] == 2
        assert comp["O"] == 2


class TestMolecularQubitEstimation:
    def test_gaba_qubits(self):
        """GABA (C4H9NO2) → 4*10 + 9*1 + 1*10 + 2*10 = 79 spin orbitals."""
        est = estimate_molecular_qubits(NEUROTRANSMITTERS["gaba"])
        assert est.n_spin_orbitals == 79

    def test_all_neurotransmitters_positive(self):
        """All NTs should have positive qubit counts."""
        for key, spec in NEUROTRANSMITTERS.items():
            est = estimate_molecular_qubits(spec)
            assert est.n_spin_orbitals > 0, f"{key} has 0 qubits"


class TestFullResourceTable:
    def test_table_completeness(self):
        """Table should cover all systems."""
        neural, molecular = full_resource_table()
        assert len(neural) == len(NEURAL_SYSTEMS)
        assert len(molecular) == len(NEUROTRANSMITTERS)
