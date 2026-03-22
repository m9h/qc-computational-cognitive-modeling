#!/bin/bash
# Batch-process multiple NWB files from a DANDI dataset
# Run on a Brev instance after setup.
#
# Usage:
#   bash scripts/run_dandi_batch.sh 001603 results/  # all files in dandiset 001603
#   bash scripts/run_dandi_batch.sh 001603 results/ 5  # first 5 files only

set -euo pipefail

DANDISET_ID="${1:?Usage: run_dandi_batch.sh <dandiset_id> <output_dir> [max_files]}"
OUTPUT_DIR="${2:?Usage: run_dandi_batch.sh <dandiset_id> <output_dir> [max_files]}"
MAX_FILES="${3:-999}"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo " QCCCM Batch Pipeline — DANDI $DANDISET_ID"
echo "============================================"

# List NWB files in the dandiset
echo ">>> Listing NWB files..."
NWB_FILES=$(uv run python -c "
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
ds = client.get_dandiset('$DANDISET_ID', 'draft')
count = 0
for asset in ds.get_assets():
    if asset.path.endswith('.nwb'):
        print(asset.path)
        count += 1
        if count >= $MAX_FILES:
            break
" 2>/dev/null)

if [ -z "$NWB_FILES" ]; then
    echo "ERROR: No NWB files found in dandiset $DANDISET_ID"
    echo "Check: https://dandiarchive.org/dandiset/$DANDISET_ID"
    exit 1
fi

N_FILES=$(echo "$NWB_FILES" | wc -l | tr -d ' ')
echo "Found $N_FILES NWB files to process"
echo ""

# Process each file
PROCESSED=0
FAILED=0

while IFS= read -r ASSET_PATH; do
    PROCESSED=$((PROCESSED + 1))
    # Clean filename for output
    SAFE_NAME=$(echo "$ASSET_PATH" | tr '/' '_' | sed 's/.nwb$//')
    OUTPUT_FILE="$OUTPUT_DIR/${SAFE_NAME}_quantum.json"

    echo "[$PROCESSED/$N_FILES] $ASSET_PATH"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "  → Already processed, skipping"
        continue
    fi

    uv run python scripts/run_dandi_pipeline.py \
        --dandiset "$DANDISET_ID" \
        --asset "$ASSET_PATH" \
        --n-neurons 4 \
        --n-samples 20 \
        --output "$OUTPUT_FILE" 2>&1 | sed 's/^/  → /' || {
            echo "  → FAILED"
            FAILED=$((FAILED + 1))
        }
done <<< "$NWB_FILES"

echo ""
echo "============================================"
echo " Batch complete: $PROCESSED processed, $FAILED failed"
echo " Results in: $OUTPUT_DIR/"
echo "============================================"
