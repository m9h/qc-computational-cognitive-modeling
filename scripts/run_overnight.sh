#!/usr/bin/env bash
# Run autoresearch overnight on DGX Spark
# Usage: bash ~/dev/quantum-cognition/scripts/run_overnight.sh

set -euo pipefail

# Source bashrc for API keys (nohup doesn't inherit interactive shell env)
source ~/.bashrc 2>/dev/null || true

cd ~/dev/quantum-cognition
git pull origin master

# Clear stale results and tree from previous runs to start fresh
rm -f autoresearch/results.tsv
rm -f autoresearch/tree.json

# agentsciml lives in its own venv at ~/dev/agentsciml
AGENTSCIML=~/dev/agentsciml/.venv/bin/agentsciml

if [ ! -x "$AGENTSCIML" ]; then
    echo "ERROR: agentsciml not found at $AGENTSCIML"
    echo "Install: cd ~/dev/agentsciml && uv sync"
    exit 1
fi

# Run agentsciml in background, pointing at this project
nohup "$AGENTSCIML" -v run \
    --project ~/dev/quantum-cognition \
    --budget 10.0 \
    --generations 10 \
    > /tmp/agentsciml_overnight.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/agentsciml_overnight.log"
echo "Monitor: tail -f /tmp/agentsciml_overnight.log"
