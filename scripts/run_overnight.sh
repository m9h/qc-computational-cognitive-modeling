#!/usr/bin/env bash
# Run autoresearch overnight on DGX Spark
# Usage: ssh 192.168.108.72 'bash ~/dev/quantum-cognition/scripts/run_overnight.sh'

set -euo pipefail

cd ~/dev/quantum-cognition
git pull origin master

# Clear stale results from previous buggy runs
rm -f autoresearch/results.tsv

# Run agentsciml in background
nohup uv run agentsciml -v run \
    --project . \
    --budget 5.0 \
    --generations 10 \
    > /tmp/agentsciml_overnight.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/agentsciml_overnight.log"
echo "Monitor: tail -f /tmp/agentsciml_overnight.log"
