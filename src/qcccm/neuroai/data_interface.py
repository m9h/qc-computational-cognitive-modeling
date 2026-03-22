"""Interface between neural recording data and quantum representations.

Bridges BL-1 cortical culture simulator / NWB organoid recordings
with QCCCM's density matrix formalism. Converts spike trains, firing
rates, and functional connectivity into quantum states for analysis
with quantum information-theoretic measures.

The pipeline:
    spike trains → firing rates → density matrix → quantum observables
                 → correlations ↗

Supports:
- BL-1 simulation output (spike rasters, numpy arrays)
- NWB spike train format (list of spike time arrays)
- Raw firing rate arrays
- Functional connectivity matrices (cross-correlation, transfer entropy)

The key insight: a neural population's joint firing statistics map
naturally to a density matrix where:
- Diagonal: P(firing pattern) from independent neurons
- Off-diagonal: correlations / coherences between neurons
- Von Neumann entropy: total uncertainty in the population state
- Mutual information: shared information between subpopulations

References:
    Schneidman, E. et al. (2006). Weak pairwise correlations imply
        strongly correlated network states. Nature 440:1007-1012.
    Sharf, T. et al. (2022). Functional neuronal circuitry and
        oscillatory dynamics in human brain organoids. Nat Commun 13:4403.
    Kagan, B.J. et al. (2022). In vitro neurons learn and exhibit
        sentience when embodied in a simulated game-world. Neuron.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from qcccm.core.density_matrix import (
    fidelity,
    purity,
)
from qcccm.neuroai.neural_states import (
    firing_rates_to_density_matrix,
    neural_entropy,
)


# ---------------------------------------------------------------------------
# Spike train → firing rates
# ---------------------------------------------------------------------------


def spike_trains_to_rates(
    spike_times: Sequence[np.ndarray],
    duration_s: float,
    bin_size_s: float = 0.050,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spike time lists to binned firing rates.

    Args:
        spike_times: list of (n_spikes,) arrays, one per unit.
        duration_s: total recording duration in seconds.
        bin_size_s: time bin width in seconds (default 50 ms).

    Returns:
        rates: (n_bins, n_units) firing rates in Hz.
        bin_edges: (n_bins + 1,) bin edge times in seconds.
    """
    n_units = len(spike_times)
    bin_edges = np.arange(0, duration_s + bin_size_s, bin_size_s)
    n_bins = len(bin_edges) - 1
    rates = np.zeros((n_bins, n_units))

    for i, times in enumerate(spike_times):
        counts, _ = np.histogram(times, bins=bin_edges)
        rates[:, i] = counts / bin_size_s  # Hz

    return rates, bin_edges


def spike_raster_to_rates(
    raster: np.ndarray,
    dt_ms: float = 0.5,
    bin_ms: float = 50.0,
) -> np.ndarray:
    """Convert BL-1 spike raster to binned firing rates.

    Args:
        raster: (T, N) binary spike raster from BL-1 simulation.
        dt_ms: simulation timestep in ms.
        bin_ms: desired bin width in ms.

    Returns:
        (n_bins, N) firing rates normalised to [0, 1] per bin.
    """
    raster = np.asarray(raster, dtype=np.float32)
    steps_per_bin = max(int(round(bin_ms / dt_ms)), 1)
    T, N = raster.shape
    n_bins = T // steps_per_bin

    if n_bins == 0:
        return np.empty((0, N), dtype=np.float64)

    trimmed = raster[:n_bins * steps_per_bin]
    binned = trimmed.reshape(n_bins, steps_per_bin, N).sum(axis=1)
    # Normalise to [0, 1]: max possible spikes per bin = steps_per_bin
    rates = binned / steps_per_bin
    return rates


# ---------------------------------------------------------------------------
# Firing rates → pairwise correlations
# ---------------------------------------------------------------------------


def compute_correlations(rates: np.ndarray) -> np.ndarray:
    """Compute pairwise Pearson correlations from firing rate time series.

    Args:
        rates: (n_bins, N) firing rates over time.

    Returns:
        (N, N) correlation matrix with diagonal = 0.
    """
    if rates.shape[0] < 2:
        N = rates.shape[1]
        return np.zeros((N, N))

    corr = np.corrcoef(rates.T)
    np.fill_diagonal(corr, 0.0)
    # Replace NaN (constant firing) with 0
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


# ---------------------------------------------------------------------------
# Neural data → density matrix pipeline
# ---------------------------------------------------------------------------


def neural_data_to_density_matrix(
    rates: np.ndarray,
    neuron_indices: Sequence[int] | None = None,
    time_bin: int = -1,
    include_correlations: bool = True,
) -> Array:
    """Convert neural recording data to a density matrix.

    The full pipeline: select neurons, extract rates at a time bin,
    compute correlations, construct density matrix.

    Args:
        rates: (n_bins, N) firing rate array.
        neuron_indices: which neurons to include (max 8 for exact
            density matrix). If None, uses first min(N, 6).
        time_bin: which time bin to use (-1 for last).
        include_correlations: whether to add off-diagonal coherences
            from pairwise correlations.

    Returns:
        (2^n, 2^n) density matrix for the selected neural subpopulation.
    """
    N = rates.shape[1]

    if neuron_indices is None:
        neuron_indices = list(range(min(N, 6)))

    n = len(neuron_indices)
    if n > 8:
        raise ValueError(
            f"Cannot construct exact density matrix for {n} neurons "
            f"(2^{n} = {2**n} dimensional). Use neuron_indices to select ≤ 8."
        )

    # Extract rates for selected neurons at the given time bin
    selected_rates = rates[time_bin, neuron_indices]
    # Clip to [epsilon, 1-epsilon] for valid density matrices
    selected_rates = np.clip(selected_rates, 0.01, 0.99)

    correlations = None
    if include_correlations and rates.shape[0] > 2:
        full_corr = compute_correlations(rates)
        correlations = full_corr[np.ix_(neuron_indices, neuron_indices)]

    return firing_rates_to_density_matrix(
        jnp.array(selected_rates),
        correlations=jnp.array(correlations) if correlations is not None else None,
    )


# ---------------------------------------------------------------------------
# Quantum observables from neural data
# ---------------------------------------------------------------------------


def quantum_neural_analysis(
    rates: np.ndarray,
    neuron_indices: Sequence[int] | None = None,
    time_bins: Sequence[int] | None = None,
) -> dict:
    """Compute quantum information-theoretic observables from neural data.

    For each time bin, constructs a density matrix and computes:
    - Von Neumann entropy (total uncertainty)
    - Purity (how "classical" vs "quantum" the state is)
    - Decoded firing rates and correlations (roundtrip verification)

    Args:
        rates: (n_bins, N) firing rate array.
        neuron_indices: which neurons to analyse (max 8).
        time_bins: which time bins to analyse. If None, uses 10
            evenly spaced bins.

    Returns:
        Dict with keys: time_bins, entropy, purity, rates_decoded,
        correlations_decoded, density_matrices.
    """
    n_total_bins = rates.shape[0]

    if time_bins is None:
        time_bins = np.linspace(0, n_total_bins - 1, min(10, n_total_bins), dtype=int).tolist()

    if neuron_indices is None:
        neuron_indices = list(range(min(rates.shape[1], 6)))

    n = len(neuron_indices)
    results = {
        "time_bins": time_bins,
        "neuron_indices": neuron_indices,
        "n_neurons": n,
        "entropy": [],
        "purity": [],
        "density_matrices": [],
    }

    for t in time_bins:
        rho = neural_data_to_density_matrix(rates, neuron_indices, time_bin=t)
        results["entropy"].append(float(neural_entropy(rho)))
        results["purity"].append(float(purity(rho)))
        results["density_matrices"].append(rho)

    results["entropy"] = np.array(results["entropy"])
    results["purity"] = np.array(results["purity"])

    return results


def neural_state_fidelity_over_time(
    rates: np.ndarray,
    neuron_indices: Sequence[int] | None = None,
    bin_step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Track how the quantum neural state changes over time.

    Computes fidelity F(ρ_t, ρ_{t+1}) between consecutive time bins.
    High fidelity = stable population state. Low fidelity = rapid change
    (e.g., during bursts or state transitions).

    Args:
        rates: (n_bins, N) firing rate array.
        neuron_indices: which neurons to track.
        bin_step: step between compared time bins.

    Returns:
        times: (n_pairs,) time bin indices.
        fidelities: (n_pairs,) fidelity values in [0, 1].
    """
    n_bins = rates.shape[0]

    if neuron_indices is None:
        neuron_indices = list(range(min(rates.shape[1], 6)))

    times_list = []
    fids_list = []

    for t in range(0, n_bins - bin_step, bin_step):
        rho_t = neural_data_to_density_matrix(rates, neuron_indices, time_bin=t)
        rho_next = neural_data_to_density_matrix(rates, neuron_indices, time_bin=t + bin_step)
        f = float(fidelity(rho_t, rho_next))
        times_list.append(t)
        fids_list.append(f)

    return np.array(times_list), np.array(fids_list)
