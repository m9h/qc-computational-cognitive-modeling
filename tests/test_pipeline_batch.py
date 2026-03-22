"""TDD tests for batch pipeline utilities."""

import json
import os
import tempfile

import numpy as np

from qcccm.pipeline.dandi import (
    PipelineConfig,
    results_to_dict,
    run_quantum_pipeline,
)


class TestBatchOutputFormat:
    """Batch pipeline must produce files in a consistent format."""

    def test_output_is_valid_json(self):
        """Each output file must be valid JSON."""
        data = _make_spike_data(4, 1.0)
        config = PipelineConfig(n_neurons=2, n_time_samples=2)
        result = run_quantum_pipeline(data, config=config)
        output = results_to_dict(result)

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(output, f)
            path = f.name

        try:
            with open(path) as f:
                recovered = json.load(f)
            assert "entropy" in recovered
            assert "purity" in recovered
        finally:
            os.unlink(path)

    def test_output_includes_metadata(self):
        """Output must include pipeline config and data metadata."""
        data = _make_spike_data(4, 1.0)
        config = PipelineConfig(n_neurons=2, n_time_samples=2)
        result = run_quantum_pipeline(data, config=config)
        output = results_to_dict(result)

        assert "config" in output
        assert "n_units_total" in output
        assert "duration_s" in output

    def test_multiple_runs_independent(self):
        """Different input data should produce different results."""
        data1 = _make_spike_data(4, 1.0, seed=1)
        data2 = _make_spike_data(4, 1.0, seed=99)
        config = PipelineConfig(n_neurons=2, n_time_samples=2)

        r1 = run_quantum_pipeline(data1, config=config)
        r2 = run_quantum_pipeline(data2, config=config)

        # Results should differ (different spike data)
        assert not np.array_equal(r1["entropy"], r2["entropy"])


def _make_spike_data(n_units, duration_s, seed=42):
    rng = np.random.RandomState(seed)
    spike_times = []
    for _ in range(n_units):
        n = rng.poisson(10 * duration_s)
        spike_times.append(np.sort(rng.uniform(0, duration_s, n)))
    return {"spike_times": spike_times, "duration_s": duration_s, "n_units": n_units}
