#!/usr/bin/env python3
"""
Weights & Biases logging daemon.
Reads JSON-formatted metric lines from stdin and logs them to W&B.

Usage (started automatically by WandbLogger):
    python3 scripts/wandb_daemon.py --project <project> --name <run_name>

Protocol:
    Each line on stdin must be valid JSON with a "step" field, e.g.:
        {"step": 5, "reward/mean_episode": 12.3, "train/actor_loss": 0.01}
    Send {"__finish__": true} to cleanly finish the run.
"""

import sys
import json
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--name",    default=None,  help="W&B run name")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("[wandb_daemon] ERROR: wandb is not installed. "
              "Run: pip install wandb", file=sys.stderr, flush=True)
        # Drain stdin so the C++ popen pipe doesn't block
        for _ in sys.stdin:
            pass
        return

    run = wandb.init(project=args.project, name=args.name, reinit=True)
    print(f"[wandb_daemon] Run started: {run.url}", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[wandb_daemon] JSON parse error: {exc} — line: {line!r}",
                  file=sys.stderr, flush=True)
            continue

        if data.get("__finish__"):
            break

        step = data.pop("step", None)
        wandb.log(data, step=step)

    wandb.finish()
    print("[wandb_daemon] Run finished.", flush=True)


if __name__ == "__main__":
    main()
