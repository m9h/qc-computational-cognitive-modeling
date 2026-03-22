"""DANDI Archive → QCCCM quantum neural analysis pipeline.

Streams NWB spike train data from DANDI, converts to density matrices,
and computes quantum information-theoretic observables. Designed for
Dendro cloud compute or local execution.

Pipeline stages:
    1. validate_spike_data: check input format
    2. spike_trains_to_rates: bin spike times into firing rates
    3. select_neurons: choose most informative subpopulation (≤8)
    4. neural_data_to_density_matrix: rates → ρ with correlations
    5. quantum_neural_analysis: entropy, purity over time
    6. neural_state_fidelity_over_time: state change detection
    7. results_to_dict: serialize for output

DANDI datasets supported:
    - 001603: van der Molen 2025 (organoid protosequences)
    - 001747: Braingeneers SpikeCanvas (organoid HD-MEA)
    - 001611: Mayama rat cortical HD-MEA
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from qcccm.neuroai.data_interface import (
    neural_state_fidelity_over_time,
    quantum_neural_analysis,
    spike_trains_to_rates,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PipelineConfig(NamedTuple):
    """Configuration for the quantum analysis pipeline."""

    n_neurons: int = 6  # neurons to select (max 8)
    bin_size_s: float = 0.050  # spike binning window (50 ms)
    n_time_samples: int = 10  # time bins to analyse
    include_correlations: bool = True  # add off-diagonal coherences
    min_firing_rate_hz: float = 0.5  # exclude silent neurons


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_spike_data(data: dict) -> bool:
    """Validate that spike data dict has required fields.

    Args:
        data: dict with spike_times, duration_s, n_units.

    Returns:
        True if valid.

    Raises:
        ValueError: if data is missing required fields or has invalid values.
    """
    if "spike_times" not in data:
        raise ValueError("Missing required key 'spike_times'")

    if data.get("duration_s", 0) <= 0:
        raise ValueError("duration_s must be positive")

    n_units = data.get("n_units", len(data.get("spike_times", [])))
    if n_units == 0 or len(data["spike_times"]) == 0:
        raise ValueError("No units found — spike_times is empty")

    return True


# ---------------------------------------------------------------------------
# Neuron selection
# ---------------------------------------------------------------------------


def select_neurons(
    rates: np.ndarray,
    n_select: int = 6,
    min_rate: float = 0.01,
) -> list[int]:
    """Select the most informative neurons for density matrix analysis.

    Strategy: exclude silent neurons, then rank by firing rate variance
    (most variable neurons carry the most information).

    Args:
        rates: (n_bins, N) firing rate array.
        n_select: target number of neurons.
        min_rate: minimum mean rate to include.

    Returns:
        List of neuron indices, length min(n_select, 8, n_active).
    """
    n_select = min(n_select, 8)  # density matrix limit

    mean_rates = rates.mean(axis=0)
    variances = rates.var(axis=0)

    # Exclude silent neurons
    active_mask = mean_rates > min_rate
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        # Fallback: just take the least-silent neurons
        active_indices = np.argsort(mean_rates)[-n_select:]

    # Rank by variance (most informative)
    active_variances = variances[active_indices]
    ranked = active_indices[np.argsort(active_variances)[::-1]]

    return ranked[:n_select].tolist()


# ---------------------------------------------------------------------------
# DANDI URL construction
# ---------------------------------------------------------------------------


def dandiset_api_url(dandiset_id: str, version: str = "draft") -> str:
    """Construct the DANDI API URL for a dandiset.

    Args:
        dandiset_id: e.g., "001603"
        version: "draft" or specific version.

    Returns:
        API URL string.
    """
    return f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/{version}/"


def asset_s3_url(dandiset_id: str, asset_path: str, version: str = "draft") -> str:
    """Construct a URL for a specific asset in a DANDI dandiset.

    Args:
        dandiset_id: e.g., "001603"
        asset_path: path within the dandiset, e.g., "sub-01/sub-01_ecephys.nwb"
        version: dandiset version.

    Returns:
        Asset URL string.
    """
    base = f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}"
    return f"{base}/versions/{version}/assets/?path={asset_path}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_quantum_pipeline(
    spike_data: dict,
    config: PipelineConfig | None = None,
) -> dict:
    """Run the full quantum neural analysis pipeline.

    Args:
        spike_data: dict with spike_times (list of arrays), duration_s, n_units.
        config: pipeline configuration. Uses defaults if None.

    Returns:
        Dict with keys: entropy, purity, fidelity_trajectory,
        neuron_indices, n_time_bins, density_matrices, config.
    """
    if config is None:
        config = PipelineConfig()

    validate_spike_data(spike_data)

    # Stage 1: Bin spike trains into firing rates
    rates, bin_edges = spike_trains_to_rates(
        spike_data["spike_times"],
        spike_data["duration_s"],
        bin_size_s=config.bin_size_s,
    )

    # Stage 2: Select neurons
    neuron_indices = select_neurons(
        rates,
        n_select=config.n_neurons,
        min_rate=config.min_firing_rate_hz * config.bin_size_s,
    )

    # Stage 3: Quantum analysis
    n_bins = rates.shape[0]
    time_bins = np.linspace(
        0, n_bins - 1, min(config.n_time_samples, n_bins), dtype=int,
    ).tolist()

    analysis = quantum_neural_analysis(
        rates,
        neuron_indices=neuron_indices,
        time_bins=time_bins,
    )

    # Stage 4: Fidelity trajectory
    times_fid, fidelities = neural_state_fidelity_over_time(
        rates, neuron_indices=neuron_indices,
    )

    return {
        "entropy": analysis["entropy"],
        "purity": analysis["purity"],
        "fidelity_trajectory": fidelities,
        "fidelity_times": times_fid,
        "neuron_indices": neuron_indices,
        "n_time_bins": len(time_bins),
        "time_bins": time_bins,
        "density_matrices": analysis["density_matrices"],
        "config": config._asdict(),
        "n_units_total": spike_data.get("n_units", len(spike_data["spike_times"])),
        "duration_s": spike_data["duration_s"],
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def results_to_dict(result: dict) -> dict:
    """Convert pipeline results to a JSON-serializable dict.

    Converts numpy arrays to lists and removes non-serializable objects
    (density matrices are converted to magnitude arrays).

    Args:
        result: dict from run_quantum_pipeline.

    Returns:
        JSON-serializable dict.
    """
    output = {}
    for key, val in result.items():
        if key == "density_matrices":
            # Convert complex density matrices to magnitude arrays
            output[key] = [np.abs(np.asarray(rho)).tolist() for rho in val]
        elif isinstance(val, np.ndarray):
            output[key] = val.tolist()
        elif isinstance(val, dict):
            output[key] = val
        elif isinstance(val, list):
            output[key] = [
                v.tolist() if isinstance(v, np.ndarray) else v for v in val
            ]
        else:
            output[key] = val

    return output


# ---------------------------------------------------------------------------
# NWB streaming (requires pynwb + fsspec, optional)
# ---------------------------------------------------------------------------


def stream_nwb_from_dandi(
    dandiset_id: str,
    asset_path: str,
    version: str = "draft",
) -> dict:
    """Stream an NWB file from DANDI and extract spike trains.

    Requires: pynwb, fsspec, h5py, dandi

    Args:
        dandiset_id: e.g., "001603"
        asset_path: path within the dandiset.
        version: dandiset version.

    Returns:
        Spike data dict compatible with run_quantum_pipeline.
    """
    try:
        from dandi.dandiapi import DandiAPIClient
        import fsspec
        import h5py
        from pynwb import NWBHDF5IO
    except ImportError:
        raise ImportError(
            "Streaming NWB from DANDI requires: pynwb, fsspec, h5py, dandi\n"
            "Install with: uv pip install pynwb fsspec h5py dandi"
        ) from None

    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id, version)
    asset = dandiset.get_asset_by_path(asset_path)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    fs = fsspec.filesystem("http")
    f = fs.open(s3_url, "rb")
    h5file = h5py.File(f, "r")
    io = NWBHDF5IO(file=h5file, mode="r", load_namespaces=True)
    nwbfile = io.read()

    # Extract spike trains from Units table
    units = nwbfile.units
    if units is None:
        raise ValueError(f"No units table found in {asset_path}")

    spike_times = []
    for i in range(len(units)):
        times = units["spike_times"][i]
        spike_times.append(np.asarray(times))

    # Duration from session
    if nwbfile.timestamps_reference_time and nwbfile.session_start_time:
        duration_s = float(max(
            max(t[-1] for t in spike_times if len(t) > 0),
            0.0,
        ))
    else:
        duration_s = float(max(t[-1] for t in spike_times if len(t) > 0))

    return {
        "spike_times": spike_times,
        "duration_s": duration_s,
        "n_units": len(spike_times),
        "metadata": {
            "dandiset_id": dandiset_id,
            "asset_path": asset_path,
            "session_description": str(nwbfile.session_description or ""),
        },
    }
