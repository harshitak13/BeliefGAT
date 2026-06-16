from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Online trainer for BeliefGAT+CityGAT.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["warmup", "train", "eval"], default="train")
    parser.add_argument("--load_offline", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--llm_enabled", default="True")
    parser.add_argument("--llm_backend", default="llama_31_8b")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--ablation", default="V5_full")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--flow_file", default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
