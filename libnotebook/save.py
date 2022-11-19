import pathlib
import pickle
from multiprocessing import cpu_count

import pandas as pd

from meltria.loader import DatasetRecord
from tsdr import tsdr

DATA_DIR = pathlib.Path("./data")


def run_tsdr(records: list[DatasetRecord]) -> list[tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    list_of_record_and_reduced_df: list = []
    for record in records:
        # run tsdr
        reducer = tsdr.Tsdr(
            "residual_integral",
            **{
                "step1_residual_integral_threshold": 20,
                "step1_residual_integral_change_start_point": False,
                "step1_residual_integral_change_start_point_n_sigma": 3,
                "step2_clustering_method_name": "dbscan",
                "step2_dbscan_min_pts": 2,
                "step2_dbscan_dist_type": "sbd",
                "step2_dbscan_algorithm": "hdbscan",
                "step2_clustering_series_type": "raw",
                "step2_clustering_choice_method": "medoid",
            },
        )
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


def load_tsdr(dataset_id: str) -> list[tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    results = []
    parent_path = DATA_DIR / f"tsdr_{dataset_id}"
    for path in parent_path.iterdir():
        with (path / "record.pkl").open("rb") as f:
            record = pickle.load(f)
        with (path / "filtered_df.pkl").open("rb") as f:
            filtered_df = pickle.load(f)
        with (path / "anomalous_df.pkl").open("rb") as f:
            anomalous_df = pickle.load(f)
        with (path / "reduced_df.pkl").open("rb") as f:
            reduced_df = pickle.load(f)
        results.append((record, filtered_df, anomalous_df, reduced_df))
    return results


def save_tsdr(
    dataset_id: str,
    record: DatasetRecord,
    filtered_df: pd.DataFrame,
    anomalous_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
) -> None:
    path = DATA_DIR / f"tsdr_{dataset_id}" / record.chaos_case_full().replace("/", "_")
    path.mkdir(parents=True)
    for obj, name in (
        (record, "record"),
        (filtered_df, "filtered_df"),
        (anomalous_df, "anomalous_df"),
        (reduced_df, "reduced_df"),
    ):
        with open(path / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
