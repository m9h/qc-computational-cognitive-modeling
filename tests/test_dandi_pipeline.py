"""TDD tests for the DANDI → QCCCM quantum analysis pipeline.

Tests are written FIRST, then the pipeline is implemented to pass them.
NWB/DANDI connections are mocked — pipeline logic is tested with synthetic
data matching the NWB spike train format.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test 1: Spike data format validation
# ---------------------------------------------------------------------------


class TestValidateSpikeData:
    """The pipeline must validate input data before processing."""

    def test_valid_spike_data_passes(self):
        from qcccm.pipeline.dandi import validate_spike_data

        data = {
            "spike_times": [np.array([0.1, 0.5, 1.2]), np.array([0.3, 0.8])],
            "duration_s": 2.0,
            "n_units": 2,
        }
        assert validate_spike_data(data) is True

    def test_missing_spike_times_raises(self):
        from qcccm.pipeline.dandi import validate_spike_data

        with pytest.raises(ValueError, match="spike_times"):
            validate_spike_data({"duration_s": 1.0, "n_units": 1})

    def test_zero_duration_raises(self):
        from qcccm.pipeline.dandi import validate_spike_data

        data = {
            "spike_times": [np.array([0.1])],
            "duration_s": 0.0,
            "n_units": 1,
        }
        with pytest.raises(ValueError, match="duration"):
            validate_spike_data(data)

    def test_empty_units_raises(self):
        from qcccm.pipeline.dandi import validate_spike_data

        data = {
            "spike_times": [],
            "duration_s": 1.0,
            "n_units": 0,
        }
        with pytest.raises(ValueError, match="units"):
            validate_spike_data(data)


# ---------------------------------------------------------------------------
# Test 2: Neuron selection heuristic
# ---------------------------------------------------------------------------


class TestSelectNeurons:
    """Pipeline must auto-select informative neurons for density matrix."""

    def test_selects_correct_count(self):
        from qcccm.pipeline.dandi import select_neurons

        rates = np.random.rand(100, 20) * 0.5  # 20 neurons, 100 bins
        indices = select_neurons(rates, n_select=6)
        assert len(indices) == 6

    def test_excludes_silent_neurons(self):
        from qcccm.pipeline.dandi import select_neurons

        rates = np.random.rand(100, 10) * 0.5
        rates[:, 3] = 0.0  # neuron 3 is silent
        rates[:, 7] = 0.0  # neuron 7 is silent
        indices = select_neurons(rates, n_select=6)
        assert 3 not in indices
        assert 7 not in indices

    def test_selects_most_variable(self):
        from qcccm.pipeline.dandi import select_neurons

        rates = np.ones((100, 10)) * 0.3  # all constant
        rates[:, 2] = np.random.rand(100)  # neuron 2 is variable
        rates[:, 5] = np.random.rand(100)  # neuron 5 is variable
        indices = select_neurons(rates, n_select=2)
        assert 2 in indices
        assert 5 in indices

    def test_respects_max_8(self):
        from qcccm.pipeline.dandi import select_neurons

        rates = np.random.rand(50, 30)
        indices = select_neurons(rates, n_select=10)
        assert len(indices) <= 8  # density matrix limit


# ---------------------------------------------------------------------------
# Test 3: Full pipeline execution
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """End-to-end pipeline: spike data → quantum analysis results."""

    def test_returns_complete_result(self):
        from qcccm.pipeline.dandi import PipelineConfig, run_quantum_pipeline

        data = _make_synthetic_spike_data(n_units=5, duration_s=2.0)
        config = PipelineConfig(n_neurons=3, n_time_samples=3)
        result = run_quantum_pipeline(data, config=config)

        assert "entropy" in result
        assert "purity" in result
        assert "fidelity_trajectory" in result
        assert "neuron_indices" in result
        assert "n_time_bins" in result
        assert "density_matrices" in result

    def test_entropy_is_array(self):
        from qcccm.pipeline.dandi import PipelineConfig, run_quantum_pipeline

        data = _make_synthetic_spike_data(n_units=5, duration_s=2.0)
        config = PipelineConfig(n_neurons=3, n_time_samples=3)
        result = run_quantum_pipeline(data, config=config)
        assert isinstance(result["entropy"], np.ndarray)
        assert len(result["entropy"]) > 0

    def test_purity_bounded(self):
        from qcccm.pipeline.dandi import PipelineConfig, run_quantum_pipeline

        data = _make_synthetic_spike_data(n_units=4, duration_s=2.0)
        config = PipelineConfig(n_neurons=2, n_time_samples=3)
        result = run_quantum_pipeline(data, config=config)
        assert np.all(result["purity"] >= -0.01)
        assert np.all(result["purity"] <= 1.01)

    def test_custom_params(self):
        from qcccm.pipeline.dandi import PipelineConfig, run_quantum_pipeline

        data = _make_synthetic_spike_data(n_units=6, duration_s=2.0)
        config = PipelineConfig(
            n_neurons=3,
            bin_size_s=0.1,
            n_time_samples=3,
            include_correlations=True,
        )
        result = run_quantum_pipeline(data, config=config)
        assert len(result["neuron_indices"]) == 3
        assert len(result["entropy"]) == 3


# ---------------------------------------------------------------------------
# Test 4: Result serialization
# ---------------------------------------------------------------------------


class TestSerializeResults:
    """Results must be serializable for saving / Dendro output."""

    def test_to_dict_is_json_compatible(self):
        import json
        from qcccm.pipeline.dandi import PipelineConfig, run_quantum_pipeline, results_to_dict

        data = _make_synthetic_spike_data(n_units=4, duration_s=1.0)
        config = PipelineConfig(n_neurons=2, n_time_samples=2)
        result = run_quantum_pipeline(data, config=config)
        d = results_to_dict(result)
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_roundtrip_preserves_keys(self):
        import json
        from qcccm.pipeline.dandi import PipelineConfig, run_quantum_pipeline, results_to_dict

        data = _make_synthetic_spike_data(n_units=4, duration_s=1.0)
        config = PipelineConfig(n_neurons=2, n_time_samples=2)
        result = run_quantum_pipeline(data, config=config)
        d = results_to_dict(result)
        recovered = json.loads(json.dumps(d))
        assert set(recovered.keys()) == set(d.keys())


# ---------------------------------------------------------------------------
# Test 5: DANDI URL construction
# ---------------------------------------------------------------------------


class TestDandiURLs:
    """Pipeline must construct correct DANDI streaming URLs."""

    def test_dandiset_url(self):
        from qcccm.pipeline.dandi import dandiset_api_url

        url = dandiset_api_url("001603")
        assert "001603" in url
        assert "api.dandiarchive.org" in url

    def test_asset_s3_url(self):
        from qcccm.pipeline.dandi import asset_s3_url

        url = asset_s3_url("001603", "sub-01/sub-01_ecephys.nwb")
        assert "001603" in url
        assert "sub-01" in url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_spike_data(
    n_units: int = 10,
    duration_s: float = 5.0,
    mean_rate_hz: float = 10.0,
    seed: int = 42,
) -> dict:
    """Create synthetic spike data matching NWB format."""
    rng = np.random.RandomState(seed)
    spike_times = []
    for _ in range(n_units):
        n_spikes = rng.poisson(mean_rate_hz * duration_s)
        times = np.sort(rng.uniform(0, duration_s, n_spikes))
        spike_times.append(times)
    return {
        "spike_times": spike_times,
        "duration_s": duration_s,
        "n_units": n_units,
    }
