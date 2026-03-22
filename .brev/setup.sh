#!/bin/bash
# .brev/setup.sh — Auto-runs when a Brev instance starts
# Brev auto-detects this file and runs it on instance creation.
#
# Usage:
#   brev create qcccm-dandi -g "nvidia-l4:1"
#   (Brev auto-clones the repo and runs this script)

set -euo pipefail

echo "============================================"
echo " QCCCM + DANDI Pipeline — Brev Auto-Setup"
echo "============================================"

# ── 1. Install uv ──────────────────────────────────────────────────
echo ">>> Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ── 2. Install QCCCM ──────────────────────────────────────────────
echo ">>> Installing QCCCM..."
QCCCM_DIR="${BREV_WORKSPACE:-$HOME/qc-computational-cognitive-modeling}"
if [ -d "$QCCCM_DIR" ]; then
    cd "$QCCCM_DIR"
else
    cd "$HOME"
    git clone https://github.com/m9h/qc-computational-cognitive-modeling.git
    cd "$HOME/qc-computational-cognitive-modeling"
fi
uv sync --group dev
echo "QCCCM installed."

# ── 3. Install DANDI streaming dependencies ────────────────────────
echo ">>> Installing DANDI/NWB dependencies..."
uv pip install pynwb fsspec h5py dandi

# ── 4. Install JAX with CUDA ──────────────────────────────────────
echo ">>> Installing JAX with CUDA support..."
uv pip install --upgrade "jax[cuda12]" 2>/dev/null || \
    uv pip install --upgrade "jax[cuda12_pip]" 2>/dev/null || \
    echo "NOTE: JAX CUDA auto-install failed. Check manually."

# ── 5. Verify ──────────────────────────────────────────────────────
echo ">>> Verifying setup..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: not detected yet"

uv run python -c "
import jax
print(f'JAX {jax.__version__} — devices: {jax.devices()}')
import qcccm
print(f'QCCCM loaded ({qcccm.__version__})')
" 2>/dev/null || echo "Verification deferred."

echo ""
echo "============================================"
echo " Auto-setup complete!"
echo ""
echo " Run the DANDI quantum pipeline:"
echo "   cd $QCCCM_DIR"
echo "   uv run python scripts/run_dandi_pipeline.py \\"
echo "     --dandiset 001603 \\"
echo "     --asset sub-OrganoidB-6mo/sub-OrganoidB-6mo_ecephys.nwb \\"
echo "     --n-neurons 4 --n-samples 20"
echo ""
echo " Or test with synthetic data:"
echo "   uv run python scripts/run_dandi_pipeline.py --synthetic"
echo ""
echo " Run tests:"
echo "   uv run pytest -v"
echo "============================================"
