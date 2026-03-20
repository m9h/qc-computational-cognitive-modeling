"""Smoke tests for the benchmark suite."""

from qcccm.benchmarks.timing import BenchmarkResult, time_fn


class TestTimeFn:
    def test_returns_positive_times(self):
        """time_fn should return positive timing values."""
        def simple(x):
            return x + 1

        jit_t, mean_t, std_t = time_fn(simple, 42, n_repeats=3)
        assert jit_t > 0
        assert mean_t >= 0
        assert std_t >= 0


class TestBenchmarks:
    def test_jit_compilation_runs(self):
        """JIT compilation benchmark should produce a BenchmarkResult."""
        from qcccm.benchmarks.bench_jax import bench_jit_compilation

        result = bench_jit_compilation()
        assert isinstance(result, BenchmarkResult)
        assert len(result.times_mean) > 0

    def test_walk_scaling_runs(self):
        """Walk scaling with small sizes should complete."""
        from qcccm.benchmarks.bench_jax import bench_walk_scaling

        result = bench_walk_scaling(site_sizes=[21, 31])
        assert isinstance(result, BenchmarkResult)
        assert len(result.sizes) == 2

    def test_network_evolution_runs(self):
        """Network evolution benchmark should complete."""
        from qcccm.benchmarks.bench_networks import bench_network_evolution

        result = bench_network_evolution(agent_counts=[3, 5])
        assert isinstance(result, BenchmarkResult)
        assert len(result.sizes) == 2
