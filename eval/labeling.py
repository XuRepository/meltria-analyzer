from functools import partial
from typing import Any

import pandas as pd

from eval.groundtruth import check_route, select_ground_truth_metrics_in_routes
from meltria.dataset import DatasetRecord
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule
from tsdr.tsdr import filter_out_no_change_metrics


def check_causal_path_based_validity(record: DatasetRecord, anomalies: pd.Series) -> bool:
    anomaly_metrics: list[str] = anomalies.filter([True]).index.to_list()
    gt_metrics_routes = select_ground_truth_metrics_in_routes(
        record.pk,
        anomaly_metrics,
        record.chaos_type(),
        record.chaos_comp(),
    )
    return any([check_route(anomaly_metrics, gt_route)[0] for gt_route, _ in gt_metrics_routes])


def detect_anomalies_in_record(
    record: DatasetRecord,
    labbeling: dict[str, dict[str, Any]],
    fi_time: int,
) -> pd.DataFrame:
    """Detect anomalies in a dataset record"""
    # Filter out metrics with no change
    filtered_df: pd.DataFrame = filter_out_no_change_metrics(record.data_df, parallel=False)

    def detect_anomaly(X: pd.Series, n_sigma: int) -> bool:
        return detect_with_n_sigma_rule(X, test_start_time=fi_time, sigma_threshold=n_sigma).size > 0

    items: list[dict[str, str | int | float]] = []
    for n_sigma in labbeling["n_sigma_rule"]["n_sigmas"]:
        anomalies: pd.Series[bool] = record.data_df.apply(partial(detect_anomaly, n_sigma=n_sigma))
        # Validate causal-path-based labeling
        valid_ok: bool = check_causal_path_based_validity(record, anomalies)
        vc: pd.Series = anomalies.value_counts()
        items.append(
            {
                "target_app": record.target_app(),
                "chaos_type": record.chaos_type(),
                "chaos_comp": record.chaos_comp(),
                "chaos_case_num": record.chaos_case_num(),
                "n_sigma": n_sigma,
                "valid_dataset_ok": valid_ok,
                "num_anomalies": f"{vc[True]}/{filtered_df.shape[1]}/{vc[True] + vc[False]}",
                "anomalies_rate": round(vc[True] / filtered_df.shape[1], 2),
            }
        )
    df = pd.DataFrame(
        items,
        columns=[
            "target_app",
            "chaos_type",
            "chaos_comp",
            "chaos_case_num",
            "n_sigma",
            "valid_dataset_ok",
            "num_anomalies",
            "anomalies_rate",
        ],
    ).set_index(["target_app", "chaos_type", "chaos_comp", "chaos_case_num", "n_sigma"])
    return df
