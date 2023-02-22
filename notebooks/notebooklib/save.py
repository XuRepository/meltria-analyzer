import pathlib
import pickle
from multiprocessing import cpu_count
from typing import Any, Final

import pandas as pd

from meltria.loader import DatasetRecord
from meltria.metric_types import ALL_METRIC_TYPES, METRIC_PREFIX_TO_TYPE
from tsdr import tsdr

DATA_DIR = pathlib.Path(__file__).parent.parent / "dataset" / "data"

tsdr_default_options: Final[dict[str, Any]] = {
    "step1_residual_integral_threshold": 20,
    "step1_residual_integral_change_start_point": False,
    "step1_residual_integral_change_start_point_n_sigma": 3,
    "step2_clustering_method_name": "dbscan",
    "step2_dbscan_min_pts": 2,
    "step2_dbscan_dist_type": "sbd",
    "step2_dbscan_algorithm": "hdbscan",
    "step2_clustering_series_type": "raw",
    "step2_clustering_choice_method": "medoid",
}


def run_tsdr(
    records: list[DatasetRecord],
    tsdr_options: dict[str, Any] = tsdr_default_options,
) -> list[tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    list_of_record_and_reduced_df: list = []
    tsdr_options = dict(tsdr_default_options, **tsdr_options)
    for record in records:
        # run tsdr
        reducer = tsdr.Tsdr("residual_integral", **tsdr_options)
        tsdr_stat, _, _ = reducer.run(
            X=record.data_df,
            pk=record.pk,
            max_workers=cpu_count(),
        )
        filtered_df: pd.DataFrame = tsdr_stat[1][0]  # simple filtered-out data
        reduced_df = tsdr_stat[-1][0]
        anomalous_df = tsdr_stat[-2][0]
        list_of_record_and_reduced_df.append((record, filtered_df, anomalous_df, reduced_df))
    return list_of_record_and_reduced_df


def _filter_metrics_by_metric_type(df: pd.DataFrame, metric_types: dict[str, bool]) -> pd.DataFrame:
    return df[
        [
            metric_name
            for metric_name in df.columns.tolist()
            for metric_type, is_selected in metric_types.items()
            if is_selected and metric_name.startswith(METRIC_PREFIX_TO_TYPE[metric_type])
        ]
    ]


def load_tsdr(
    dataset_id: str,
    revert_normalized_time_series: bool = False,
    suffix: str = "",
    metric_types: dict[str, bool] = ALL_METRIC_TYPES,
) -> list[tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    dir_name: str = f"tsdr_{dataset_id}" if suffix == "" else f"tsdr_{dataset_id}_{suffix}"
    results = []
    parent_path = DATA_DIR / dir_name
    for path in parent_path.iterdir():
        with (path / "record.pkl").open("rb") as f:
            record = pickle.load(f)
        with (path / "filtered_df.pkl").open("rb") as f:
            filtered_df = _filter_metrics_by_metric_type(pickle.load(f), metric_types)
        with (path / "anomalous_df.pkl").open("rb") as f:
            anomalous_df = _filter_metrics_by_metric_type(pickle.load(f), metric_types)
        with (path / "reduced_df.pkl").open("rb") as f:
            reduced_df = _filter_metrics_by_metric_type(pickle.load(f), metric_types)
            if revert_normalized_time_series:  # Workaround
                for metric_name, _ in reduced_df.items():
                    reduced_df[metric_name] = anomalous_df[metric_name]
        results.append((record, filtered_df, anomalous_df, reduced_df))
    return results


def save_tsdr(
    dataset_id: str,
    record: DatasetRecord,
    filtered_df: pd.DataFrame,
    anomalous_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
    suffix: str = "",
) -> None:
    dir_name: str = f"tsdr_{dataset_id}" if suffix == "" else f"tsdr_{dataset_id}_{suffix}"
    path = DATA_DIR / dir_name / record.chaos_case_full().replace("/", "_")
    path.mkdir(parents=True, exist_ok=True)
    for obj, name in (
        (record, "record"),
        (filtered_df, "filtered_df"),
        (anomalous_df, "anomalous_df"),
        (reduced_df, "reduced_df"),
    ):
        with open(path / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
