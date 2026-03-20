"""Entry point: python -m qcccm.benchmarks [--suite {all,jax,networks}]"""

from __future__ import annotations

import argparse

from qcccm.benchmarks import (
    bench_density_matrix_ops,
    bench_jit_compilation,
    bench_network_evolution,
    bench_network_observables,
    bench_walk_scaling,
    print_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="QCCCM Benchmarks")
    parser.add_argument(
        "--suite",
        choices=["all", "jax", "networks"],
        default="all",
        help="Which benchmark suite to run",
    )
    args = parser.parse_args()

    results = []

    if args.suite in ("all", "jax"):
        print("Running JAX benchmarks...")
        results.append(bench_jit_compilation())
        results.append(bench_walk_scaling())
        results.append(bench_density_matrix_ops())

    if args.suite in ("all", "networks"):
        print("Running network benchmarks...")
        results.append(bench_network_evolution())
        results.append(bench_network_observables())

    print_results(results)


if __name__ == "__main__":
    main()
