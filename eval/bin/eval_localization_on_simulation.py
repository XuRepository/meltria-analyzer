#!/usr/bin/env python
import argparse
import pathlib
import sys

from simulation.feature_reduction import REDUCTION_METHODS
from simulation.localization import sweep_reduction_and_localization

pyrca_path = pathlib.Path(__file__).parent.parent.parent / "PyRCA"
sys.path.append(str(pyrca_path))
result_path = pathlib.Path(__file__).parent.parent.parent / "notebooks" / "materials"

num_trials = 5
anomaly_types = [0, 1]
func_types = ["identity"]  # "square", "sin", "tanh"]
noise_types = ["normal", "exponential", "uniform", "laplace"]
weight_generators = ["normal", "uniform"]
data_scale_params = [
    {"num_node": 50, "num_edge": 100, "num_normal_samples": 160, "num_abnormal_samples": 20},
    {"num_node": 50, "num_edge": 200, "num_normal_samples": 160, "num_abnormal_samples": 20},
    {"num_node": 100, "num_edge": 500, "num_normal_samples": 160, "num_abnormal_samples": 20},
    {"num_node": 100, "num_edge": 750, "num_normal_samples": 160, "num_abnormal_samples": 20},
]


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval_localization_on_simulation")
    parser.add_argument("--resuming-trial-no", type=int, required=False, default=1, help="number of trials for resuming")
    parser.add_argument("--trial-no", type=int, required=False, default=-1, help="target trial number")
    parser.add_argument("--rcd-seed-ensemble", type=int, required=False, default=-1, help="number of RCD seed")
    parser.add_argument("--reduction-methods", type=str, nargs="*", required=False, default=REDUCTION_METHODS + ["Ideal"], help="reduction method")
    args = parser.parse_args()

    if not result_path.is_dir():
        raise FileNotFoundError(f"{result_path} does not exist.")

    method_suffix = ("_" + "-".join(args.reduction_methods)) if args.reduction_methods else ""

    if args.trial_no > 0:
        if args.rcd_seed_ensemble > 0:
            ldf = sweep_reduction_and_localization(
                anomaly_types, data_scale_params, func_types, noise_types, weight_generators, [args.trial_no],
                localization_methods=["RCD"],
                reduction_methods=args.reduction_methods,
                localization_kwargs={"rcd_n_iters": args.rcd_seed_ensemble},
                n_jobs=1,
            )
            ldf.to_pickle(
                result_path / f"pyrca_feature_reduction_simulation_localization_results_{args.trial_no}_rcd.pkl.gz", compression="gzip",
            )
        else:
            ldf = sweep_reduction_and_localization(
                anomaly_types, data_scale_params, func_types, noise_types, weight_generators, [args.trial_no],
                reduction_methods=args.reduction_methods,
                n_jobs=parser.njobs,
            )
            ldf.to_pickle(
                result_path / f"pyrca_feature_reduction_simulation_localization_results_{args.trial_no}{method_suffix}.pkl.gz", compression="gzip",
            )
    else:
        for trial_no in range(args.resuming_trial_no, num_trials + 1):
            if args.rcd_seed_ensemble > 0:
                ldf = sweep_reduction_and_localization(
                    anomaly_types, data_scale_params, func_types, noise_types, weight_generators, [trial_no], localization_methods=["RCD"], reduction_methods=args.reduction_methods,
                    n_jobs=1,
                    rcd_n_iters=args.rcd_seed_ensemble,
                )
                ldf.to_pickle(
                    result_path / f"pyrca_feature_reduction_simulation_localization_results_{trial_no}_rcd.pkl.gz", compression="gzip",
                )
            else:
                ldf = sweep_reduction_and_localization(
                    anomaly_types, data_scale_params, func_types, noise_types, weight_generators, [trial_no],
                    reduction_methods=args.reduction_methods,
                    n_jobs=-1,
                )
                ldf.to_pickle(
                    result_path / f"pyrca_feature_reduction_simulation_localization_results_{trial_no}{method_suffix}.pkl.gz", compression="gzip",
                )


if __name__ == "__main__":
    main()
