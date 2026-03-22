"""Qubit resource estimation for neural system simulation.

Implements the scaling analysis from Wolff et al. (2025) Fig. 3:
maps biological systems to qubit requirements under different
quantum encoding schemes.

Three encoding strategies:
1. Amplitude encoding: ⌈log₂ N⌉ qubits for N neurons (exponential compression)
2. Basis state encoding: N qubits for N neurons (one qubit per neuron)
3. One-hot encoding: N qubits, only one active at a time

For molecular/neurotransmitter simulation:
- Spin-orbital mapping: Σ (atoms × spin_orbitals_per_atom) qubits

References:
    Wolff, A. et al. (2025). Quantum Computing for Neuroscience.
    Babbush, R. et al. (2018). Low-depth quantum simulation of materials.
"""

from __future__ import annotations

import math
from typing import NamedTuple



class NeuralSystemSpec(NamedTuple):
    """Specification of a neural system for qubit estimation."""

    name: str
    n_neurons: int
    description: str = ""


class QubitEstimate(NamedTuple):
    """Qubit requirements under different encoding schemes."""

    system: str
    n_neurons: int
    amplitude_qubits: int  # ⌈log₂ N⌉
    basis_state_qubits: int  # N
    notes: str = ""


class MolecularSpec(NamedTuple):
    """Specification of a molecule for quantum chemistry qubit estimation."""

    name: str
    formula: str
    description: str = ""


class MolecularQubitEstimate(NamedTuple):
    """Qubit requirements for molecular quantum chemistry."""

    name: str
    formula: str
    n_atoms: int
    n_spin_orbitals: int  # qubits needed
    notes: str = ""


# ---------------------------------------------------------------------------
# Standard biological systems (from Wolff et al. 2025 Fig. 3)
# ---------------------------------------------------------------------------

NEURAL_SYSTEMS: dict[str, NeuralSystemSpec] = {
    "cortical_minicolumn": NeuralSystemSpec(
        name="Cortical Minicolumn",
        n_neurons=90,
        description="Buxhoeveden & Casanova, Brain 125:935-951 (2002)",
    ),
    "c_elegans": NeuralSystemSpec(
        name="C. elegans",
        n_neurons=302,
        description="Cook et al., Genes Dev 33:592-608 (2019)",
    ),
    "drosophila": NeuralSystemSpec(
        name="Drosophila melanogaster",
        n_neurons=199_380,
        description="Raji & Potter, PNAS 118:e2020709118 (2021)",
    ),
    "mouse": NeuralSystemSpec(
        name="Mouse",
        n_neurons=71_000_000,
        description="Herculano-Houzel et al., PNAS 103:12138-43 (2006)",
    ),
    "human": NeuralSystemSpec(
        name="Human",
        n_neurons=86_000_000_000,
        description="Azevedo et al., J Comp Neurol 513:532-541 (2009)",
    ),
    "human_dyad": NeuralSystemSpec(
        name="Human Dyad",
        n_neurons=172_000_000_000,
        description="Two interacting human brains (Dumas hyperscanning)",
    ),
    # Organoid / culture systems relevant to BL-1
    "organoid_small": NeuralSystemSpec(
        name="Small Organoid (DIV 60)",
        n_neurons=10_000,
        description="~10K neurons, early cortical organoid",
    ),
    "organoid_mature": NeuralSystemSpec(
        name="Mature Organoid (DIV 180+)",
        n_neurons=100_000,
        description="~100K neurons, Sharf et al. 2022",
    ),
    "dissociated_culture": NeuralSystemSpec(
        name="Dissociated Culture (MEA)",
        n_neurons=50_000,
        description="~50K neurons on 60ch MEA, Wagenaar et al. 2006",
    ),
    "hd_mea_recording": NeuralSystemSpec(
        name="HD-MEA Recording",
        n_neurons=1_000,
        description="~1K neurons recorded simultaneously, MaxOne 26K electrodes",
    ),
}


# Spin orbitals per atom (STO-3G basis, from Wolff et al.)
SPIN_ORBITALS_PER_ATOM: dict[str, int] = {
    "H": 1, "C": 10, "N": 10, "O": 10, "P": 10, "S": 10,
}

NEUROTRANSMITTERS: dict[str, MolecularSpec] = {
    "glycine": MolecularSpec("Glycine", "NH2CH2COOH", "Simplest amino acid NT"),
    "gaba": MolecularSpec("GABA", "C4H9NO2", "Primary inhibitory NT"),
    "serotonin": MolecularSpec("5-HT", "C10H12N2O", "Monoamine NT"),
    "dopamine": MolecularSpec("DA", "C8H11NO2", "Catecholamine NT"),
    "acetylcholine": MolecularSpec("ACh", "C7H16NO2", "Cholinergic NT"),
    "glutamate": MolecularSpec("Glu", "C5H9NO4", "Primary excitatory NT"),
}


# ---------------------------------------------------------------------------
# Estimation functions
# ---------------------------------------------------------------------------


def estimate_neural_qubits(spec: NeuralSystemSpec) -> QubitEstimate:
    """Estimate qubits needed to simulate a neural system.

    Args:
        spec: neural system specification.

    Returns:
        QubitEstimate with amplitude and basis state requirements.
    """
    n = spec.n_neurons
    amp = math.ceil(math.log2(max(n, 2)))
    return QubitEstimate(
        system=spec.name,
        n_neurons=n,
        amplitude_qubits=amp,
        basis_state_qubits=n,
        notes=spec.description,
    )


def parse_molecular_formula(formula: str) -> dict[str, int]:
    """Parse a molecular formula into element counts.

    Handles formulas with repeated elements (e.g., NH2CH2COOH has C×2).

    Args:
        formula: e.g., "C4H9NO2" or "NH2CH2COOH"

    Returns:
        Dict mapping element symbol to total count.
    """
    import re
    result: dict[str, int] = {}
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if element:
            result[element] = result.get(element, 0) + (int(count) if count else 1)
    return result


def estimate_molecular_qubits(spec: MolecularSpec) -> MolecularQubitEstimate:
    """Estimate qubits for quantum chemistry simulation of a molecule.

    Uses the spin-orbital mapping: each atom contributes a fixed number
    of spin orbitals based on the STO-3G basis set.

    Args:
        spec: molecular specification.

    Returns:
        MolecularQubitEstimate with atom count and qubit requirement.
    """
    composition = parse_molecular_formula(spec.formula)
    n_atoms = sum(composition.values())
    n_spin_orbitals = sum(
        count * SPIN_ORBITALS_PER_ATOM.get(element, 10)
        for element, count in composition.items()
    )
    return MolecularQubitEstimate(
        name=spec.name,
        formula=spec.formula,
        n_atoms=n_atoms,
        n_spin_orbitals=n_spin_orbitals,
    )


def full_resource_table() -> tuple[list[QubitEstimate], list[MolecularQubitEstimate]]:
    """Generate the complete resource estimation table.

    Reproduces Wolff et al. (2025) Fig. 3 data.

    Returns:
        (neural_estimates, molecular_estimates)
    """
    neural = [estimate_neural_qubits(spec) for spec in NEURAL_SYSTEMS.values()]
    molecular = [estimate_molecular_qubits(spec) for spec in NEUROTRANSMITTERS.values()]
    return neural, molecular


def print_resource_table() -> None:
    """Print a formatted resource estimation table."""
    neural, molecular = full_resource_table()

    print("Neural Systems — Qubit Requirements")
    print("=" * 65)
    print(f"{'System':<25s} {'Neurons':>15s} {'Amp. qubits':>12s}")
    print("-" * 65)
    for est in neural:
        print(f"{est.system:<25s} {est.n_neurons:>15,d} {est.amplitude_qubits:>12d}")

    print("\nNeurotransmitters — Spin-Orbital Qubits")
    print("=" * 55)
    print(f"{'Name':<12s} {'Formula':<15s} {'Atoms':>6s} {'Qubits':>8s}")
    print("-" * 55)
    for est in molecular:
        print(f"{est.name:<12s} {est.formula:<15s} {est.n_atoms:>6d} {est.n_spin_orbitals:>8d}")
