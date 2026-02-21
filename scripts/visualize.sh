#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# visualize.sh — Load a saved checkpoint and open the MuJoCo visualizer.
#
# Usage:
#   ./scripts/visualize.sh <checkpoint>
#
# Example:
#   ./scripts/visualize.sh checkpoints_7/network_100.pt
#
# The window requires a display. Inside a dev container you need X11 forwarding:
#   - SSH:  ssh -X user@host  then run VS Code / terminal from there
#   - On the host machine:  just run this script directly
#
# Interactive controls (inside the window):
#   R            — reset pendulum to starting position (angle = π)
#   →  / ←       — push the pendulum right / left
#   Ctrl + RMB   — reset (mouse)
#   Ctrl + LMB   — push (mouse)
#   Mouse drag   — rotate camera
#   Scroll       — zoom
#   ESC          — exit
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$REPO_ROOT/build/bin/visualize"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path_to_checkpoint>"
    echo ""
    echo "Available checkpoints:"
    find "$REPO_ROOT" -name "network_*.pt" -o -name "actor_*.pt" 2>/dev/null \
        | sort -V | sed 's|'"$REPO_ROOT/"'||'
    exit 1
fi

CHECKPOINT="$1"

# Resolve relative paths from repo root
if [[ ! -f "$CHECKPOINT" ]]; then
    if [[ -f "$REPO_ROOT/$CHECKPOINT" ]]; then
        CHECKPOINT="$REPO_ROOT/$CHECKPOINT"
    else
        echo "Error: checkpoint not found: $CHECKPOINT"
        exit 1
    fi
fi

# Check the binary exists; offer to build if not
if [[ ! -f "$BINARY" ]]; then
    echo "Visualize binary not found. Building..."
    cmake -S "$REPO_ROOT" -B "$REPO_ROOT/build" -DCMAKE_BUILD_TYPE=Release -Wno-dev
    cmake --build "$REPO_ROOT/build" --target visualize -- -j"$(nproc)"
fi

# Check a display is available
if [[ -z "${DISPLAY:-}" ]]; then
    echo ""
    echo "WARNING: \$DISPLAY is not set — the window may fail to open."
    echo ""
    echo "If you are inside a dev container, enable X11 forwarding:"
    echo "  • Connect via: ssh -X <user>@<host>  or use VS Code Remote with X11"
    echo "  • Then re-run this script"
    echo ""
    read -r -p "Try anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 0
fi

echo "Loading checkpoint: $CHECKPOINT"
echo "Opening MuJoCo visualizer..."
echo ""
echo "Controls:"
echo "  R           reset to starting position"
echo "  →  ←        push pendulum right / left"
echo "  Mouse drag  rotate camera"
echo "  Scroll      zoom"
echo "  ESC         quit"
echo ""

"$BINARY" "$CHECKPOINT"
