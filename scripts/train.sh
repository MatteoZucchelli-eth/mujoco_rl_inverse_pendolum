#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train.sh — Build and launch training inside a detached tmux session.
#
# Usage:
#   ./scripts/train.sh [options]
#
# Options:
#   -c, --checkpoint <path>   Resume from a saved network_N.pt
#   -p, --project    <name>   W&B project name  (enables W&B logging)
#   -r, --run        <name>   W&B run name       (optional)
#   -s, --session    <name>   tmux session name  (default: training)
#   --no-build                Skip cmake/make step
#   -h, --help                Show this message
#
# Examples:
#   ./scripts/train.sh                                        # fresh start
#   ./scripts/train.sh -p inverse-pendulum -r exp-01         # with W&B
#   ./scripts/train.sh -c checkpoints_7/network_100.pt -p inverse-pendulum
#   ./scripts/train.sh --no-build -p inverse-pendulum        # skip rebuild
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
BINARY="$BUILD_DIR/bin/mujoco_rl"
SESSION="training"
CHECKPOINT=""
WANDB_PROJECT=""
WANDB_RUN=""
DO_BUILD=true

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -p|--project)    WANDB_PROJECT="$2"; shift 2 ;;
        -r|--run)        WANDB_RUN="$2"; shift 2 ;;
        -s|--session)    SESSION="$2"; shift 2 ;;
        --no-build)      DO_BUILD=false; shift ;;
        -h|--help)
            sed -n '2,/^# ----/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------- sanity checks ----------
if [[ -n "$CHECKPOINT" && ! -f "$CHECKPOINT" ]]; then
    echo "Error: checkpoint not found: $CHECKPOINT"
    exit 1
fi

# ---------- build ----------
if $DO_BUILD; then
    echo ">>> Building..."
    mkdir -p "$BUILD_DIR"
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Wno-dev
    cmake --build "$BUILD_DIR" --target mujoco_rl -- -j"$(nproc)"
    echo ">>> Build done."
fi

# ---------- assemble launch command ----------
CMD="$BINARY"
[[ -n "$CHECKPOINT"    ]] && CMD+=" --checkpoint \"$CHECKPOINT\""
[[ -n "$WANDB_PROJECT" ]] && CMD+=" --project \"$WANDB_PROJECT\""
[[ -n "$WANDB_RUN"     ]] && CMD+=" --run \"$WANDB_RUN\""

# Redirect output to a log file as well so you can read it without attaching
LOG_FILE="$REPO_ROOT/training.log"
FULL_CMD="cd \"$REPO_ROOT\" && $CMD 2>&1 | tee \"$LOG_FILE\""

# ---------- tmux session ----------
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "A tmux session named '$SESSION' already exists."
    echo "  Attach : tmux attach -t $SESSION"
    echo "  Kill   : tmux kill-session -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION" bash
tmux send-keys -t "$SESSION" "$FULL_CMD" Enter

echo ""
echo "Training started in tmux session '$SESSION'."
echo ""
echo "  Attach (live output) : tmux attach -t $SESSION"
echo "  Detach once attached : Ctrl+B  then  D"
echo "  Follow log           : tail -f $LOG_FILE"
echo "  Kill session         : tmux kill-session -t $SESSION"
echo ""
if [[ -n "$WANDB_PROJECT" ]]; then
    echo "  W&B project : $WANDB_PROJECT"
    [[ -n "$WANDB_RUN" ]] && echo "  W&B run     : $WANDB_RUN"
    echo "  Make sure you have run: wandb login"
fi
