#!/usr/bin/env python3
"""Run the QCCCM quantum neural analysis pipeline on DANDI data.

Usage:
    # Stream from DANDI and analyse:
    python scripts/run_dandi_pipeline.py --dandiset 001603 --asset sub-01/sub-01_ecephys.nwb

    # Analyse a local NWB file:
    python scripts/run_dandi_pipeline.py --local path/to/recording.nwb

    # Run on synthetic data (for testing):
    python scripts/run_dandi_pipeline.py --synthetic --n-units 10 --duration 5.0

    # Customise analysis:
    python scripts/run_dandi_pipeline.py --synthetic --n-neurons 4 --bin-size 0.1 --n-samples 10

Output:
    JSON results saved to --output (default: qcccm_results.json)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from qcccm.pipeline.dandi import (
    PipelineConfig,
    results_to_dict,
    run_quantum_pipeline,
    validate_spike_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QCCCM quantum neural analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dandiset", type=str, help="DANDI dataset ID (e.g., 001603)")
    source.add_argument("--local", type=str, help="Path to local NWB file")
    source.add_argument("--synthetic", action="store_true", help="Use synthetic data")

    # DANDI options
    parser.add_argument("--asset", type=str, help="Asset path within DANDI dataset")
    parser.add_argument("--version", type=str, default="draft", help="DANDI version")

    # Synthetic options
    parser.add_argument("--n-units", type=int, default=10, help="Synthetic: number of units")
    parser.add_argument("--duration", type=float, default=5.0, help="Synthetic: duration (s)")

    # Pipeline config
    parser.add_argument("--n-neurons", type=int, default=4, help="Neurons for density matrix")
    parser.add_argument("--bin-size", type=float, default=0.05, help="Bin size (s)")
    parser.add_argument("--n-samples", type=int, default=10, help="Time samples to analyse")

    # Output
    parser.add_argument("--output", type=str, default="qcccm_results.json", help="Output JSON path")

    args = parser.parse_args()

    config = PipelineConfig(
        n_neurons=args.n_neurons,
        bin_size_s=args.bin_size,
        n_time_samples=args.n_samples,
    )

    # Load data
    if args.synthetic:
        print(f"Generating synthetic data: {args.n_units} units, {args.duration}s")
        rng = np.random.RandomState(42)
        spike_times = []
        for _ in range(args.n_units):
            n_spikes = rng.poisson(10.0 * args.duration)
            times = np.sort(rng.uniform(0, args.duration, n_spikes))
            spike_times.append(times)
        spike_data = {
            "spike_times": spike_times,
            "duration_s": args.duration,
            "n_units": args.n_units,
        }

    elif args.local:
        print(f"Loading local NWB: {args.local}")
        try:
            from bl1.validation.loaders import load_nwb_spike_trains
            spike_data = load_nwb_spike_trains(args.local)
        except ImportError:
            print("bl1 not installed. Trying pynwb directly...")
            from qcccm.pipeline.dandi import stream_nwb_from_dandi
            raise NotImplementedError("Direct NWB loading not yet implemented. Install bl1.")

    elif args.dandiset:
        if not args.asset:
            print("Error: --asset is required with --dandiset", file=sys.stderr)
            sys.exit(1)
        print(f"Streaming from DANDI: {args.dandiset}/{args.asset}")
        from qcccm.pipeline.dandi import stream_nwb_from_dandi
        spike_data = stream_nwb_from_dandi(args.dandiset, args.asset, args.version)

    # Validate
    validate_spike_data(spike_data)
    print(f"Data: {spike_data['n_units']} units, {spike_data['duration_s']:.1f}s")

    # Run pipeline
    print(f"Running quantum analysis (n_neurons={config.n_neurons}, "
          f"bin={config.bin_size_s}s, samples={config.n_time_samples})...")
    result = run_quantum_pipeline(spike_data, config=config)

    # Report
    print("\nResults:")
    print(f"  Neurons selected: {result['neuron_indices']}")
    print(f"  Entropy range: [{result['entropy'].min():.3f}, {result['entropy'].max():.3f}]")
    print(f"  Purity range:  [{result['purity'].min():.3f}, {result['purity'].max():.3f}]")
    if len(result['fidelity_trajectory']) > 0:
        print(f"  Mean fidelity: {result['fidelity_trajectory'].mean():.3f}")

    # Save
    output = results_to_dict(result)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
