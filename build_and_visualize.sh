#!/bin/bash
set -e

mkdir -p build && cd build
cmake ..
make -j8

# Find the latest actor checkpoint
CHECKPOINT=$(ls ../checkpoints_7/actor_*.pt 2>/dev/null | sort -V | tail -n 1)

if [ -z "$CHECKPOINT" ]; then
    echo "No actor checkpoint found in ../checkpoints_3/"
    exit 1
fi

echo "Running visualization with checkpoint: $CHECKPOINT"
./bin/visualize "$CHECKPOINT"
