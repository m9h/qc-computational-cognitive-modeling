#!/bin/bash
# Setup script for RunPod GPU instance
#
# Launch a RunPod pod with:
#   - Template: RunPod PyTorch (or any Ubuntu-based)
#   - GPU: RTX 4090 ($0.39/hr) or A40 ($0.39/hr) or A100 ($1.64/hr)
#   - Disk: 20GB
#
# Then SSH in and run:
#   curl -fsSL https://raw.githubusercontent.com/m9h/qc-computational-cognitive-modeling/main/scripts/setup_runpod.sh | bash

set -euo pipefail

echo "============================================"
echo " QCCCM + DANDI Pipeline — RunPod Setup"
echo "============================================"

# ── 1. Install uv ─────────────────────────────────────────────
echo ">>> Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ── 2. Clone and install QCCCM ────────────────────────────────
echo ">>> Cloning QCCCM..."
cd /workspace  # RunPod persistent storage
if [ ! -d qc-computational-cognitive-modeling ]; then
    git clone https://github.com/m9h/qc-computational-cognitive-modeling.git
fi
cd qc-computational-cognitive-modeling
uv sync --group dev

# ── 3. Install DANDI + NWB streaming deps ─────────────────────
echo ">>> Installing DANDI dependencies..."
uv pip install pynwb fsspec h5py dandi scipy

# ── 4. Install JAX CUDA ───────────────────────────────────────
echo ">>> Installing JAX with CUDA..."
uv pip install --upgrade "jax[cuda12]" 2>/dev/null || \
    uv pip install --upgrade "jax[cuda12_pip]" 2>/dev/null || \
    echo "JAX CUDA install needs manual config"

# ── 5. Verify ─────────────────────────────────────────────────
echo ">>> Verifying..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
uv run python -c "
import jax
print(f'JAX {jax.__version__} — {jax.devices()}')
import qcccm
print('QCCCM loaded')
"

echo ""
echo "============================================"
echo " Setup complete! Run the pipeline:"
echo ""
echo "   cd /workspace/qc-computational-cognitive-modeling"
echo "   uv run python scripts/run_dandi_pipeline.py --synthetic"
echo ""
echo " Batch process DANDI 001747 (remaining 142 files):"
echo "   uv run python /tmp/batch_fast.py"
echo "============================================"
