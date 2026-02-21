#!/usr/bin/env bash
# Convenience wrapper — builds the visualize binary and opens the viewer
# with the latest available checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Build
cmake -S "$REPO_ROOT" -B "$REPO_ROOT/build" -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build "$REPO_ROOT/build" --target visualize -- -j"$(nproc)"

# Find latest checkpoint (network_*.pt preferred, fall back to actor_*.pt)
CHECKPOINT=$(ls "$REPO_ROOT"/checkpoints_7/network_*.pt 2>/dev/null | sort -V | tail -n 1)
if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT=$(ls "$REPO_ROOT"/checkpoints_7/actor_*.pt 2>/dev/null | sort -V | tail -n 1)
fi

if [[ -z "$CHECKPOINT" ]]; then
    echo "No checkpoint found in checkpoints_7/. Train first:"
    echo "  ./build_and_train.sh"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
exec "$REPO_ROOT/scripts/visualize.sh" "$CHECKPOINT"
