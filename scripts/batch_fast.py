"""Fast batch processing — no correlations, 3 neurons for speed.

Processes remaining DANDI 001747 files (skips already-done ones).
Also processes DANDI 001611 if time permits.

Usage:
    uv run python scripts/batch_fast.py                    # 001747 only
    uv run python scripts/batch_fast.py --include-rat      # + 001611 (2700 files)
    uv run python scripts/batch_fast.py --max-files 50     # limit count
"""

import argparse
import json
import os
import time

from dandi.dandiapi import DandiAPIClient
from qcccm.pipeline.dandi import (
    PipelineConfig,
    results_to_dict,
    run_quantum_pipeline,
    stream_nwb_from_dandi,
)


def process_dataset(ds_id, config, out_dir, max_files=9999):
    """Process all ecephys NWB files in a DANDI dataset."""
    client = DandiAPIClient()
    ds = client.get_dandiset(ds_id, "draft")

    ecephys = [
        a.path for a in ds.get_assets()
        if "ecephys" in a.path and a.path.endswith(".nwb")
    ][:max_files]

    os.makedirs(out_dir, exist_ok=True)
    print(f"\nDANDI {ds_id}: {len(ecephys)} files to process")

    processed = failed = skipped = 0

    for i, path in enumerate(ecephys):
        safe = path.replace("/", "_").replace(".nwb", "")
        out_file = f"{out_dir}/{safe}_quantum.json"

        if os.path.exists(out_file):
            skipped += 1
            continue

        t0 = time.time()
        print(f"[{i+1}/{len(ecephys)}] {path}", end=" ", flush=True)

        try:
            data = stream_nwb_from_dandi(ds_id, path)
            result = run_quantum_pipeline(data, config=config)
            output = results_to_dict(result)
            output["metadata"] = data.get("metadata", {})
            output["metadata"]["dandiset"] = ds_id
            output["metadata"]["asset_path"] = path

            with open(out_file, "w") as f:
                json.dump(output, f)

            dt = time.time() - t0
            e = result["entropy"].mean()
            print(f"OK ({dt:.0f}s) S={e:.3f}")
            processed += 1

        except Exception as e:
            dt = time.time() - t0
            print(f"FAIL ({dt:.0f}s): {e}")
            failed += 1

    print(f"\n{ds_id}: {processed} processed, {failed} failed, {skipped} skipped")
    return processed, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-rat", action="store_true", help="Include DANDI 001611")
    parser.add_argument("--max-files", type=int, default=9999)
    args = parser.parse_args()

    # Fast config: 3 neurons, no correlations overhead in the pipeline
    config = PipelineConfig(
        n_neurons=3,
        n_time_samples=10,
        bin_size_s=0.1,
        include_correlations=False,
    )

    process_dataset("001747", config, "results/001747", args.max_files)

    if args.include_rat:
        process_dataset("001611", config, "results/001611", args.max_files)

    print("\n=== BATCH COMPLETE ===")


if __name__ == "__main__":
    main()
