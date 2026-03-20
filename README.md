# QCCCM — Quantum Compute for Computational Cognitive Modeling

Bridging quantum computing and computational cognitive neuroscience. QCCCM provides JAX-native quantum primitives designed for cognitive modeling: density matrices as generalized beliefs, quantum walks as evidence accumulation, and a classical-quantum bridge connecting stochastic processes to unitary evolution via the Szegedy construction.

## Installation

```bash
uv sync           # core deps
uv sync --group dev  # + pytest, ruff, mypy
```

Optional backends:

```bash
uv pip install "qcccm[ibm]"       # Qiskit
uv pip install "qcccm[annealing]" # D-Wave
uv pip install "qcccm[mitiq]"     # error mitigation
```

## Package layout

```
src/qcccm/
  core/
    states.py          # |0⟩, |+⟩, Bell, GHZ state preparation
    density_matrix.py  # von Neumann entropy, partial trace, QMI, purity, fidelity
    quantum_walk.py    # Hadamard/biased coins, QW evolution (lax.scan), FPT density
  models/
    bridge.py          # Szegedy stochastic→unitary, beliefs↔density matrix, quantum EFE
  circuits/            # (planned) PennyLane/Qiskit circuit constructors
  fitting/             # (planned) parameter estimation for cognitive data
  networks/            # (planned) multi-agent quantum cognitive networks
  viz/
    bloch.py           # Bloch sphere visualization
    walks.py           # QW probability evolution, spreading comparison, FPT plots
notebooks/
  01_bits_to_qubits.ipynb  # Pedagogical intro — no QM prerequisites
```

## Quick start

```python
from qcccm.core import (
    computational_basis, bell_state,
    pure_state_density_matrix, von_neumann_entropy, partial_trace,
    QuantumWalkParams, quantum_walk_evolution,
)
from qcccm.models.bridge import beliefs_to_density_matrix, quantum_efe

# Density matrix from classical beliefs
import jax.numpy as jnp
rho = beliefs_to_density_matrix(jnp.array([0.7, 0.2, 0.1]))
print(f"S(rho) = {von_neumann_entropy(rho):.4f}")

# Quantum walk with ballistic spreading
params = QuantumWalkParams(n_sites=101, n_steps=50, start_pos=50)
probs = quantum_walk_evolution(params)  # (51, 101)
```

## Tests

```bash
uv run pytest -v
```

## License

MIT
