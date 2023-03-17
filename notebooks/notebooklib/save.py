import math
import pathlib
from collections import defaultdict
from multiprocessing import cpu_count
from typing import Any, Final, Generator

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

from eval.groundtruth import check_cause_metrics
from meltria.loader import DatasetRecord, is_prometheus_exporter_default_metrics
from meltria.metric_types import ALL_METRIC_TYPES, METRIC_PREFIX_TO_TYPE
from tsdr import tsdr

DATA_DIR = pathlib.Path(__file__).parent.parent / "dataset" / "data"

TSDR_DEFAULT_PHASE1_METHOD: Final[str] = "residual_integral"

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


def run_and_save_tsdr_to_each_set(
    dataset_id: str,
    records: list[DatasetRecord],
    phase1_method: str = TSDR_DEFAULT_PHASE1_METHOD,
    tsdr_options: dict[str, Any] = tsdr_default_options,
    suffix: str = "",
) -> None:
    params = {
        "only_services": {
            "services": True,
            "containers": False,
            "middlewares": False,
            "nodes": False,
        },
        "only_ctnrs": {
            "services": True,
            "containers": True,
            "middlewares": False,
            "nodes": False,
        },
        "middlewares": {
            "services": True,
            "containers": True,
            "middlewares": True,
            "nodes": False,
        },
    }
    for type_name, metric_types in params.items():
        run_and_save_tsdr(dataset_id, records, phase1_method, tsdr_options, metric_types, f"{suffix}_{type_name}")


def run_tsdr_as_generator(
    records: list[DatasetRecord],
    phase1_method: str = TSDR_DEFAULT_PHASE1_METHOD,
    tsdr_options: dict[str, Any] = tsdr_default_options,
    metric_types: dict[str, bool] = ALL_METRIC_TYPES,
) -> Generator[tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
    tsdr_options = dict(tsdr_default_options, **tsdr_options)
    for record in records:
        # run tsdr
        reducer = tsdr.Tsdr(phase1_method, **tsdr_options)
        tsdr_stat, _, _ = reducer.run(
            X=_filter_metrics_by_metric_type(
                _filter_prometheus_exporter_go_metrics(record.data_df),
                metric_types,
            ),
            pk=record.pk,
            max_workers=cpu_count(),
        )
        filtered_df: pd.DataFrame = tsdr_stat[1][0]  # simple filtered-out data
        reduced_df = tsdr_stat[-1][0]
        anomalous_df = tsdr_stat[-2][0]
        yield (record, filtered_df, anomalous_df, reduced_df)


def run_tsdr(
    records: list[DatasetRecord],
    phase1_method: str = TSDR_DEFAULT_PHASE1_METHOD,
    tsdr_options: dict[str, Any] = tsdr_default_options,
    metric_types: dict[str, bool] = ALL_METRIC_TYPES,
) -> list[tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    return list(run_tsdr_as_generator(records, phase1_method, tsdr_options, metric_types))


def run_and_save_tsdr(
    dataset_id: str,
    records: list[DatasetRecord],
    phase1_method: str = TSDR_DEFAULT_PHASE1_METHOD,
    tsdr_options: dict[str, Any] = tsdr_default_options,
    metric_types: dict[str, bool] = ALL_METRIC_TYPES,
    suffix: str = "",
) -> None:
    for record, filtered_df, anomalous_df, reduced_df in run_tsdr_as_generator(
        records, phase1_method, tsdr_options, metric_types
    ):
        save_tsdr(dataset_id, record, filtered_df, anomalous_df, reduced_df, suffix=suffix)
        del record, filtered_df, anomalous_df, reduced_df  # for memory efficiency


def _filter_metrics_by_metric_type(df: pd.DataFrame, metric_types: dict[str, bool]) -> pd.DataFrame:
    return df[
        [
            metric_name
            for metric_name in df.columns.tolist()
            for metric_type, is_selected in metric_types.items()
            if is_selected and metric_name.startswith(METRIC_PREFIX_TO_TYPE[metric_type])
        ]
    ]


def _filter_prometheus_exporter_go_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, [not is_prometheus_exporter_default_metrics(metric_name) for metric_name in df.columns.tolist()]]


def _group_by_metric_type(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for metric_type, _ in ALL_METRIC_TYPES.items():
        result[metric_type] = _filter_metrics_by_metric_type(df, {metric_type: True})
    return result


def load_tsdr_by_chaos(
    dataset_id: str,
    revert_normalized_time_series: bool = False,
    suffix: str = "",
) -> dict[
    tuple[str, str], list[tuple[DatasetRecord, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]]
]:  # (chaos_type, chaos_comp)
    datasets = load_tsdr_grouped_by_metric_type(dataset_id, revert_normalized_time_series, suffix)
    results = defaultdict(list)
    for record, df_by_metric_type in datasets:
        results[(record.chaos_type(), record.chaos_comp())].append((record, df_by_metric_type))
    return results


def load_tsdr_grouped_by_metric_type(
    dataset_id: str,
    revert_normalized_time_series: bool = False,
    suffix: str = "",
) -> list[tuple[DatasetRecord, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]]:
    dir_name: str = f"tsdr_{dataset_id}" if suffix == "" else f"tsdr_{dataset_id}_{suffix}"
    results = []
    parent_path = DATA_DIR / dir_name
    for path in parent_path.iterdir():
        if not any(path.iterdir()):  # check empty
            continue
        with (path / "record.bz2").open("rb") as f:
            record = joblib.load(f)
        with (path / "filtered_df.bz2").open("rb") as f:
            filtered_df = _group_by_metric_type(_filter_prometheus_exporter_go_metrics(joblib.load(f)))
        with (path / "anomalous_df.bz2").open("rb") as f:
            anomalous_df = _group_by_metric_type(_filter_prometheus_exporter_go_metrics(joblib.load(f)))
        with (path / "reduced_df.bz2").open("rb") as f:
            reduced_df = _group_by_metric_type(_filter_prometheus_exporter_go_metrics(joblib.load(f)))
            if revert_normalized_time_series:  # Workaround
                for metric_name, _ in reduced_df.items():
                    reduced_df[metric_name] = anomalous_df[metric_name]
        df_by_metric_type: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        for metric_type in ALL_METRIC_TYPES.keys():
            df_by_metric_type[metric_type] = (
                filtered_df[metric_type],
                anomalous_df[metric_type],
                reduced_df[metric_type],
            )
        results.append((record, df_by_metric_type))
    return results


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
        with (path / "record.bz2").open("rb") as f:
            record = joblib.load(f)
        with (path / "filtered_df.bz2").open("rb") as f:
            filtered_df = _filter_prometheus_exporter_go_metrics(
                _filter_metrics_by_metric_type(joblib.load(f), metric_types)
            )
        with (path / "anomalous_df.bz2").open("rb") as f:
            anomalous_df = _filter_prometheus_exporter_go_metrics(
                _filter_metrics_by_metric_type(joblib.load(f), metric_types)
            )
        with (path / "reduced_df.bz2").open("rb") as f:
            reduced_df = _filter_prometheus_exporter_go_metrics(
                _filter_metrics_by_metric_type(joblib.load(f), metric_types)
            )
            if revert_normalized_time_series:  # Workaround
                for metric_name, _ in reduced_df.items():
                    reduced_df[metric_name] = anomalous_df[metric_name]
        results.append((record, filtered_df, anomalous_df, reduced_df))
    return results


def save_tsdr_each_set(
    dataset_id: str,
    results: dict[str, list[tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame]]],
    suffix: str = "",
) -> None:
    for name, items in results.items():
        for item in items:
            save_tsdr(dataset_id, *item, suffix=f"{suffix}_{name}")


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
        joblib.dump(obj, path / f"{name}.bz2", compress=("bz2", 3))  # type: ignore


def validate_datasets(
    datasets: list[tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame]]
) -> tuple[pd.DataFrame, dict]:
    check_results = []
    dataset_by_chaos_case = {}
    for record, filtered_df, anomalous_df, reduced_df in datasets:
        anomalous_ok, anomalous_cause_metrics = check_cause_metrics(
            record.pk,
            anomalous_df.columns.tolist(),
            chaos_type=record.chaos_type(),
            chaos_comp=record.chaos_comp(),
            optional_cause=True,
        )
        reduced_ok, reduced_cause_metrics = check_cause_metrics(
            record.pk,
            reduced_df.columns.tolist(),
            chaos_type=record.chaos_type(),
            chaos_comp=record.chaos_comp(),
            optional_cause=True,
        )
        check_results.append(
            (
                record.target_app(),
                record.chaos_type(),
                record.chaos_comp(),
                record.chaos_case_num(),
                anomalous_ok,
                anomalous_cause_metrics,
                reduced_ok,
                reduced_cause_metrics,
            )
        )
        dataset_by_chaos_case[(record.chaos_type(), record.chaos_comp(), record.chaos_case_num())] = (
            record,
            filtered_df,
            anomalous_df,
            reduced_df,
        )
    return (
        pd.DataFrame(
            check_results,
            columns=[
                "target_app",
                "chaos_type",
                "chaos_comp",
                "chaos_case_num",
                "anomalous_ok",
                "anomalous_cause_metrics",
                "reduced_ok",
                "reduced_cause_metrics",
            ],
        )
        .set_index(["target_app", "chaos_type", "chaos_comp", "chaos_case_num"])
        .sort_index()
    ), dataset_by_chaos_case


def plot_figures_of_cause_metrics(
    validation_results: pd.DataFrame, dataset_by_chaos_case: dict, only_clustered_ng: bool = False
) -> None:
    def _plot_figure_of_cause_metrics(data_df: pd.DataFrame, row: pd.Series, title: str) -> None:
        _target_df = data_df.apply(lambda x: scipy.stats.zscore(x)).filter(regex=f"{row.chaos_comp}")
        n, ncols = 6, 3
        nrows = math.ceil((_target_df.shape[1] / n) / ncols)
        fig, axs = plt.subplots(figsize=(20, 3.0 * nrows), ncols=ncols, nrows=nrows)
        fig.suptitle(f"{title}: {row.chaos_type}/{row.chaos_comp}/{row.chaos_case_num}")
        fig.tight_layout()
        for i, ax in zip(range(0, _target_df.shape[1], n), axs.flatten()):
            for col, ts in _target_df.iloc[:, i : i + n].items():
                ax.plot(ts, label=col)
            ax.legend(loc="upper left", fontsize=8)
        plt.show()
        plt.close(fig=fig)

    for row in validation_results.reset_index().itertuples():
        if only_clustered_ng and row.reduced_ok:
            continue
        record, filtered_df, anomalous_df, reduced_df = dataset_by_chaos_case[
            row.chaos_type,
            row.chaos_comp,
            row.chaos_case_num,
        ]
        _plot_figure_of_cause_metrics(anomalous_df, row, "anomalous")
        _plot_figure_of_cause_metrics(reduced_df, row, "clustered")
