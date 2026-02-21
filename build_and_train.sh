#!/usr/bin/env bash
# Convenience wrapper — delegates to scripts/train.sh.
# All arguments are forwarded.
#
# Examples:
#   ./build_and_train.sh
#   ./build_and_train.sh -p inverse-pendulum -r run-001
#   ./build_and_train.sh -c checkpoints_7/network_100.pt -p inverse-pendulum
exec "$(dirname "$0")/scripts/train.sh" "$@"
