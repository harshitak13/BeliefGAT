import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import experiment_arg_parser, run_experiment


if __name__ == "__main__":
    run_experiment(experiment_arg_parser("Run SUMO 4x4 experiment.").parse_args())
