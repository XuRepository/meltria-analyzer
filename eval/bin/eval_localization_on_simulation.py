#!/usr/bin/env python
import argparse
import pathlib
import sys

import pandas as pd

from simulation.localization import sweep_reduction_and_localization

pyrca_path = pathlib.Path(__file__).parent.parent.parent / "PyRCA"
sys.path.append(str(pyrca_path))
result_path = pathlib.Path(__file__).parent.parent.parent / "notebooks" / "materials"

num_trials = 5
anomaly_types = [0, 1]
func_types = ["identity"]  #"square", "sin", "tanh"]
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
    args = parser.parse_args()

    if not result_path.is_dir():
        raise FileNotFoundError(f"{result_path} does not exist.")

    localization_dfs = []
    for trial_no in range(1, num_trials + 1):
        ldf = sweep_reduction_and_localization(
            anomaly_types, data_scale_params, func_types, noise_types, weight_generators, [trial_no], n_jobs=-1,
        )
        ldf.to_pickle(
            result_path / f"pyrca_feature_reduction_simulation_localization_results_{trial_no}.pkl.gz", compression="gzip",
        )

    localization_df = pd.concat(localization_dfs, ignore_index=True)
    localization_df.to_pickle(result_path / "pyrca_feature_reduction_simulation_localization_results.pkl.gz", compression="gzip")


if __name__ == "__main__":
    main()
