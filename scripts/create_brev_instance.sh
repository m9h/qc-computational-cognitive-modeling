#!/bin/bash
# Create a Brev GPU instance for QCCCM DANDI pipeline
# Run this from your local machine after: brev login
#
# Usage:
#   ./scripts/create_brev_instance.sh              # default: L4 (cheapest GPU)
#   ./scripts/create_brev_instance.sh l4            # L4 24GB — $0.16/hr spot
#   ./scripts/create_brev_instance.sh a100          # A100 80GB — for large analyses
#
# The DANDI data is on AWS us-east-2. Brev instances are also on AWS,
# so data streaming is fast and free (no egress cost).

set -euo pipefail

INSTANCE_NAME="qcccm-dandi"

GPU_CHOICE="${1:-l4}"

case "$GPU_CHOICE" in
    l4)
        GPU_FLAG="nvidia-l4:1"
        echo "GPU: NVIDIA L4 24GB (cheapest — sufficient for ≤8 qubit density matrices)"
        ;;
    l40s)
        GPU_FLAG="nvidia-l40s:1"
        echo "GPU: NVIDIA L40S 48GB"
        ;;
    a100)
        GPU_FLAG="nvidia-a100-80gb:1"
        echo "GPU: NVIDIA A100 80GB (for large-scale JAX analysis)"
        ;;
    h100)
        GPU_FLAG="nvidia-h100:1"
        echo "GPU: NVIDIA H100 96GB"
        ;;
    cpu)
        GPU_FLAG=""
        echo "CPU only (density matrix ops for ≤6 neurons are CPU-fine)"
        ;;
    *)
        echo "Unknown GPU: $GPU_CHOICE"
        echo "Options: l4 (default), l40s, a100, h100, cpu"
        exit 1
        ;;
esac

echo ""
echo "Creating Brev instance: $INSTANCE_NAME"
echo ""

if ! brev ls &> /dev/null; then
    echo "ERROR: Not authenticated. Run: brev login"
    exit 1
fi

# Create from GitHub repo — Brev auto-detects .brev/setup.sh
if [ -z "$GPU_FLAG" ]; then
    brev create "$INSTANCE_NAME" --repo https://github.com/m9h/qc-computational-cognitive-modeling
else
    brev create "$INSTANCE_NAME" -g "$GPU_FLAG" --repo https://github.com/m9h/qc-computational-cognitive-modeling
fi

echo ""
echo "Instance created. It will auto-run .brev/setup.sh"
echo ""
echo "Next steps:"
echo "  1. Wait for ready:  brev ls"
echo "  2. SSH in:          brev shell $INSTANCE_NAME"
echo "  3. Run pipeline:    uv run python scripts/run_dandi_pipeline.py --dandiset 001603 --asset <path>"
echo ""
echo "Available DANDI organoid datasets:"
echo "  001603  van der Molen 2025 — protosequences in organoids (111 NWB files)"
echo "  001747  Braingeneers SpikeCanvas — large-scale organoid HD-MEA (160 NWB files)"
echo "  001611  Mayama rat cortical HD-MEA — dissociated cultures (2700 NWB files)"
echo ""
