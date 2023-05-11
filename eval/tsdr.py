import datetime
import hashlib
import logging
import os
import pathlib
from collections import defaultdict
from multiprocessing import cpu_count
from typing import Any, Final, Generator

import joblib
import neptune
import pandas as pd

from eval.groundtruth import check_cause_metrics
from meltria.loader import DatasetRecord, is_prometheus_exporter_default_metrics
from meltria.metric_types import ALL_METRIC_TYPES, METRIC_PREFIX_TO_TYPE
from tsdr import tsdr

DEFAULT_CHAOS_TYPES: Final[set[str]] = {"pod-cpu-hog", "pod-memory-hog"}
DATA_DIR = pathlib.Path(__file__).parent.parent / "dataset" / "data"
TSDR_DEFAULT_PHASE1_METHOD: Final[str] = "residual_integral"
METRIC_TYPES_PAIRS: Final[list[dict[str, bool]]] = [
    {
        "services": True,
        "containers": True,
        "middlewares": False,
        "nodes": False,
    },
    {
        "services": True,
        "containers": True,
        "middlewares": True,
        "nodes": False,
    },
]

tsdr_default_options: Final[dict[str, Any]] = {
    "step1_method_name": "residual_integral",
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


def generate_file_path_suffix_as_id(
    tsdr_options: dict[str, Any], metric_types: dict[str, bool], time_range: tuple[int, int], **kwargs: Any
) -> str:
    return hashlib.md5(
        "".join(
            [f"{k}{v}" for k, v in sorted(tsdr_options.items())]
            + [f"{k}{v}" for k, v in sorted(metric_types.items())]
            + ([] if time_range[0] == 0 and time_range[1] == 0 else [f"time_range{time_range[0]}{time_range[1]}"])
            + [f"{k}{v}" for k, v in sorted(kwargs.items())]
        ).encode()
    ).hexdigest()


def sweep_tsdr_and_save_as_cache(
    dataset_id: str,
    records: list[DatasetRecord],
    list_of_tsdr_options: list[dict[str, Any]],
    use_manually_selected_metrics: list[bool] = [True, False],
    metric_types_pairs: list[dict[str, bool]] = METRIC_TYPES_PAIRS,
    time_range: tuple[int, int] = (0, 0),
    experiment_id: str = "",
) -> None:
    experiment_id = experiment_id or datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    for tsdr_options in list_of_tsdr_options:
        for metric_types in metric_types_pairs:
            for use_manually_selected_metric in use_manually_selected_metrics:
                run_tsdr_and_save_as_cache(
                    experiment_id=experiment_id,
                    dataset_id=dataset_id,
                    metric_types=metric_types,
                    records=records,
                    tsdr_options=tsdr_options,
                    time_range=time_range,
                    use_manually_selected_metrics=use_manually_selected_metric,
                )


def run_tsdr_and_save_as_cache(
    experiment_id: str,
    dataset_id: str,
    metric_types: dict[str, bool],
    records: list[DatasetRecord],
    tsdr_options: dict[str, Any] = tsdr_default_options,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
) -> None:
    # Workaround
    # if (
    #     metric_types["middlewares"]
    #     and tsdr_options.get("step2_dbscan_algorithm") == "dbscan"
    #     and tsdr_options.get("step2_dbscan_dist_type") == "pearsonr"
    # ):
    #     logging.info("Skip dbscan with pearsonr to dataset including middlewares because it takes too long time.")
    #     return

    file_path_suffix = generate_file_path_suffix_as_id(
        tsdr_options, metric_types, time_range=time_range, use_manually_selected_metrics=use_manually_selected_metrics
    )

    run = neptune.init_run(project=os.environ["TSDR_NEPTUNE_PROJECT"])
    run["experiment_id"] = experiment_id
    run["dataset/dataset_id"] = dataset_id
    run["dataset/target_app"] = records[0].target_app()
    run["dataset/metric_types"] = metric_types
    run["dataset/use_manually_selected_metrics"] = use_manually_selected_metrics
    run["dataset/time_range/start"] = time_range[0]
    run["dataset/time_range/end"] = time_range[1]
    run["parameters"] = tsdr_options

    score_dfs: list[pd.DataFrame] = []
    for record, filtered_df, anomalous_df, reduced_df, tsdr_stat in run_tsdr_as_generator(
        records=records,
        tsdr_options=tsdr_options,
        enable_unireducer=tsdr_options["enable_unireducer"],
        enable_multireducer=tsdr_options["enable_multireducer"],
        metric_types=metric_types,
        use_manually_selected_metrics=use_manually_selected_metrics,
        time_range=time_range,
    ):
        score_dfs.append(calculate_scores_from_tsdr_result(tsdr_stat, record, metric_types))

        save_tsdr(dataset_id, record, filtered_df, anomalous_df, reduced_df, file_path_suffix=file_path_suffix)
        del record, filtered_df, anomalous_df, reduced_df  # for memory efficiency

    upload_scores_to_neptune(run, pd.concat(score_dfs, axis=0), metric_types)
    run.stop()


def run_tsdr_as_generator(
    records: list[DatasetRecord],
    tsdr_options: dict[str, Any],
    enable_unireducer: bool = True,
    enable_multireducer: bool = True,
    metric_types: dict[str, bool] = ALL_METRIC_TYPES,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
) -> Generator[
    tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[tuple[pd.DataFrame, pd.DataFrame, float]]],
    None,
    None,
]:
    tsdr_options = dict(tsdr_default_options, **tsdr_options)

    for record in records:
        prefiltered_data_df = _filter_prometheus_exporter_go_metrics(record.data_df)
        if use_manually_selected_metrics:
            prefiltered_data_df = filter_manually_selected_metrics(prefiltered_data_df)

        start, end = time_range
        if end == 0:
            end = prefiltered_data_df.shape[0]

        reducer = tsdr.Tsdr(tsdr_options["step1_method_name"], **tsdr_options)
        tsdr_stat, _, _ = reducer.run(
            X=filter_metrics_by_metric_type(prefiltered_data_df.iloc[start:end, :], metric_types),
            pk=record.pk,
            max_workers=cpu_count(),
        )

        filtered_df: pd.DataFrame = tsdr_stat[1][0]  # simple filtered-out data
        if enable_unireducer:
            anomalous_df = tsdr_stat[2][0]
        else:
            anomalous_df = filtered_df
        if enable_multireducer:
            reduced_df = tsdr_stat[-1][0]
        else:
            reduced_df = anomalous_df
        yield (record, filtered_df, anomalous_df, reduced_df, tsdr_stat)


def _get_cause_metrics(record: DatasetRecord, metrics: list, optional_cause: bool = True) -> list[str]:
    cause_metrics_exist, found_metrics = check_cause_metrics(
        pk=record.pk,
        metrics=metrics,
        chaos_type=record.chaos_type(),
        chaos_comp=record.chaos_comp(),
        optional_cause=optional_cause,
    )
    if not cause_metrics_exist:
        logging.warning(
            f"Cause metrics not found: pk={record.pk}, chaos_type={record.chaos_type()}, chaos_comp={record.chaos_comp()}"
        )
    return found_metrics.tolist()


def recall_of_cause_metrics(total_cause_metrics: set[str], found_cause_metrics: set[str]) -> float:
    return len(total_cause_metrics & found_cause_metrics) / len(total_cause_metrics)


def proportion_of_cause_metrics(total_metrics: set[str], found_cause_metrics: set[str]) -> float:
    return len(found_cause_metrics) / len(total_metrics)


def calculate_scores_from_tsdr_result(
    tsdr_stat: list[tuple[pd.DataFrame, pd.DataFrame, float]],
    record: DatasetRecord,
    metric_types: dict[str, bool],
) -> pd.DataFrame:
    # metrics denominator after phase0 simple filtering
    total_cause_metrics: set[str] = set(_get_cause_metrics(record, list(tsdr_stat[1][0].columns)))
    total_mandatory_cause_metrics: set[str] = set(
        _get_cause_metrics(record, list(tsdr_stat[1][0].columns), optional_cause=False)
    )
    tests: list[dict[str, Any]] = []
    for i, (reduced_df, stat_df, elapsed_time) in enumerate(tsdr_stat[1:], start=1):
        cause_metrics_exist, found_metrics = check_cause_metrics(
            pk=record.pk,
            metrics=list(reduced_df.columns),
            chaos_type=record.chaos_type(),
            chaos_comp=record.chaos_comp(),
            optional_cause=True,
        )
        num_series_by_type: dict[str, int] = {}
        for metric_type, enable in metric_types.items():
            if not enable:
                continue
            num_series_by_type[f"num_series/{metric_type}/raw"] = tsdr_stat[0][1].loc[metric_type]["count"].sum()
            num_series_by_type[f"num_series/{metric_type}/filtered"] = (
                tsdr_stat[1][1].loc[metric_type]["count"].sum() if metric_type in tsdr_stat[1][1] else 0
            )
            num_series_by_type[f"num_series/{metric_type}/reduced"] = (
                stat_df.loc[metric_type]["count"].sum() if metric_type in stat_df else 0
            )
        tests.append(
            {
                "chaos_type": record.chaos_type(),
                "chaos_comp": record.chaos_comp(),
                "metrics_file": record.basename_of_metrics_file(),
                "phase": f"phase{i}",
                "cause_metrics/exist": cause_metrics_exist,
                "cause_metrics/only_mandatory_recall": recall_of_cause_metrics(
                    total_mandatory_cause_metrics, set(found_metrics)
                ),
                "cause_metrics/recall": recall_of_cause_metrics(total_cause_metrics, set(found_metrics)),
                "cause_metrics/proportion": proportion_of_cause_metrics(set(reduced_df.columns), set(found_metrics)),
                "cause_metrics/num_total": len(total_cause_metrics),
                "cause_metrics/num_found": len(found_metrics),
                "num_series/total/raw": tsdr_stat[0][1]["count"].sum(),  # raw
                "num_series/total/filtered": tsdr_stat[1][1]["count"].sum(),  # after step0
                "num_series/total/reduced": stat_df["count"].sum(),  # after step{i}
                **num_series_by_type,
                "elapsed_time": elapsed_time,
                "found_metrics": ",".join(found_metrics),
            }
        )
    return pd.DataFrame(tests).set_index(["chaos_type", "chaos_comp", "metrics_file", "phase"])


def upload_scores_to_neptune(run: neptune.Run, tests_df: pd.DataFrame, target_metric_types: dict[str, bool]) -> None:
    def agg_score(x: pd.DataFrame) -> pd.Series:
        tp = int(x["cause_metrics/exist"].sum())
        fn = int((~x["cause_metrics/exist"]).sum())
        rate = 1 - x["num_series/total/reduced"] / x["num_series/total/filtered"]
        num_series_items: dict[str, str] = {}
        for metric_type, ok in target_metric_types.items():
            if not ok:
                continue
            num_series_items[f"num_series/{metric_type}"] = "/".join(
                [
                    f"{int(x[f'num_series/{metric_type}/reduced'].mean())}",
                    f"{int(x[f'num_series/{metric_type}/filtered'].mean())}",
                    f"{int(x[f'num_series/{metric_type}/raw'].mean())}",
                ]
            )
        d = {
            "tp": tp,
            "fn": fn,
            "accuracy": tp / (tp + fn),
            "cause_metrics/recall_mean": x["cause_metrics/recall"].mean(),
            "cause_metrics/recall_mandatory_mean": x["cause_metrics/only_mandatory_recall"].mean(),
            "cause_metrics/proportion_mean": x["cause_metrics/proportion"].mean(),
            "cause_metrics/num_total_mean": x["cause_metrics/num_total"].mean(),
            "cause_metrics/num_found_mean": x["cause_metrics/num_found"].mean(),
            "num_series/total": "/".join(
                [
                    f"{int(x['num_series/total/reduced'].mean())}",
                    f"{int(x['num_series/total/filtered'].mean())}",
                    f"{int(x['num_series/total/raw'].mean())}",
                ]
            ),
            **num_series_items,
            "reduction_rate_mean": rate.mean(),
            "reduction_rate_max": rate.max(),
            "reduction_rate_min": rate.min(),
            "elapsed_time": x["elapsed_time"].mean(),
            "elapsed_time_max": x["elapsed_time"].max(),
            "elapsed_time_min": x["elapsed_time"].min(),
        }
        return pd.Series(d)

    run["scores/summary"].upload(neptune.types.File.as_html(tests_df))
    scores_by_phase = tests_df.groupby("phase").apply(agg_score).reset_index().set_index("phase")
    scores_by_chaos_type = (
        tests_df.groupby(["chaos_type", "phase"]).apply(agg_score).reset_index().set_index(["chaos_type", "phase"])
    )
    scores_by_chaos_comp = (
        tests_df.groupby(["chaos_comp", "phase"]).apply(agg_score).reset_index().set_index(["chaos_comp", "phase"])
    )
    scores_by_chaos_type_and_comp = (
        tests_df.groupby(
            ["chaos_type", "chaos_comp", "phase"],
        )
        .apply(agg_score)
        .reset_index()
        .set_index(["chaos_type", "chaos_comp", "phase"])
    )
    total_scores: pd.Series = scores_by_phase.iloc[-1, :].copy()
    for col in ["elapsed_time", "elapsed_time_max", "elapsed_time_min"]:
        total_scores[col] = scores_by_phase[col].sum()
    for col in [
        "cause_metrics/recall_mean",
        "cause_metrics/recall_mandatory_mean",
        "cause_metrics/proportion_mean",
        "cause_metrics/num_total_mean",
        "cause_metrics/num_found_mean",
    ]:
        total_scores[col] = scores_by_phase[col][-1]

    run["scores"] = total_scores.to_dict()
    run["scores/summary_by_phase"].upload(neptune.types.File.as_html(scores_by_phase))
    run["scores/summary_by_chaos_type"].upload(neptune.types.File.as_html(scores_by_chaos_type))
    run["scores/summary_by_chaos_comp"].upload(neptune.types.File.as_html(scores_by_chaos_comp))
    run["scores/summary_by_chaos_type_and_chaos_comp"].upload(
        neptune.types.File.as_html(scores_by_chaos_type_and_comp)
    )


def filter_metrics_by_metric_type(df: pd.DataFrame, metric_types: dict[str, bool]) -> pd.DataFrame:
    return df[
        [
            metric_name
            for metric_name in df.columns.tolist()
            for metric_type, is_selected in metric_types.items()
            if is_selected and metric_name.startswith(METRIC_PREFIX_TO_TYPE[metric_type])
        ]
    ]


MANUALLY_SELECTED_METRICS: Final[set[str]] = {
    "file_descriptors",
    "processes",
    "sockets",
    "threads",
    "cpu_system_seconds_total",
    "cpu_usage_seconds_total",
    "cpu_user_seconds_total",
    "memory_rss",
    "memory_usage_bytes",
    "memory_working_set_bytes",
    "fs_reads_bytes_total",
    "fs_reads_total",
    "fs_writes_bytes_total",
    "fs_writes_total",
    "network_receive_bytes_total",
    "network_receive_packets_total",
    "network_transmit_bytes_total",
    "network_transmit_packets_total",
}


def filter_manually_selected_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Filter only container metrics
    return df.loc[
        :,
        [
            metric_name.startswith("s-")
            or metric_name.startswith("n-")
            or any([metric_name.endswith(base_name) for base_name in MANUALLY_SELECTED_METRICS])
            for metric_name in df.columns.tolist()
        ],
    ]


def _filter_prometheus_exporter_go_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, [not is_prometheus_exporter_default_metrics(metric_name) for metric_name in df.columns.tolist()]]


def _group_by_metric_type(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for metric_type, _ in ALL_METRIC_TYPES.items():
        result[metric_type] = filter_metrics_by_metric_type(df, {metric_type: True})
    return result


def load_tsdr_by_chaos(
    dataset_id: str,
    metric_types: dict[str, bool],
    revert_normalized_time_series: bool = False,
    tsdr_options: dict[str, Any] = tsdr_default_options,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
    target_chaos_types: set[str] = DEFAULT_CHAOS_TYPES,
) -> dict[
    tuple[str, str], list[tuple[DatasetRecord, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]]
]:  # (chaos_type, chaos_comp)
    datasets = load_tsdr_grouped_by_metric_type(
        dataset_id,
        metric_types,
        revert_normalized_time_series,
        tsdr_options,
        use_manually_selected_metrics,
        time_range,
    )
    results = defaultdict(list)
    for record, df_by_metric_type in datasets:
        if record.chaos_type() not in target_chaos_types:
            continue
        results[(record.chaos_type(), record.chaos_comp())].append((record, df_by_metric_type))
    return results


def load_tsdr_grouped_by_metric_type(
    dataset_id: str,
    metric_types: dict[str, Any],
    revert_normalized_time_series: bool = False,
    tsdr_options: dict[str, Any] = tsdr_default_options,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
) -> list[tuple[DatasetRecord, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]]:
    ok, parent_path = check_cache_suffix(
        dataset_id,
        metric_types,
        tsdr_options,
        use_manually_selected_metrics,
        time_range,
    )
    if not ok:
        raise ValueError(
            f"Dataset {dataset_id} is not cached, {parent_path} does not exist. {metric_types}, {tsdr_options}, use_manually_selected_metrics={use_manually_selected_metrics},time_range={time_range}"
        )
    results = []
    for path in parent_path.iterdir():
        if not any(path.iterdir()):  # check empty
            continue
        with (path / "record.bz2").open("rb") as f:
            record = joblib.load(f)
            record.data_df = _filter_prometheus_exporter_go_metrics(record.data_df)
        with (path / "filtered_df.bz2").open("rb") as f:
            filtered_df = joblib.load(f)
            filtered_df = _group_by_metric_type(_filter_prometheus_exporter_go_metrics(filtered_df))
        with (path / "anomalous_df.bz2").open("rb") as f:
            anomalous_df = joblib.load(f)
            anomalous_df = _group_by_metric_type(_filter_prometheus_exporter_go_metrics(anomalous_df))
        with (path / "reduced_df.bz2").open("rb") as f:
            reduced_df = joblib.load(f)
            reduced_df = _group_by_metric_type(_filter_prometheus_exporter_go_metrics(reduced_df))
        df_by_metric_type: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        for metric_type in ALL_METRIC_TYPES.keys():
            if revert_normalized_time_series:  # Workaround
                for metric_name, _ in reduced_df[metric_type].items():
                    reduced_df[metric_type][metric_name] = anomalous_df[metric_type][metric_name]
            df_by_metric_type[metric_type] = (
                filtered_df[metric_type],
                anomalous_df[metric_type],
                reduced_df[metric_type],
            )
        results.append((record, df_by_metric_type))
    return results


def check_cache_suffix(
    dataset_id: str,
    metric_types: dict[str, Any],
    tsdr_options: dict[str, Any] = tsdr_default_options,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
) -> tuple[bool, pathlib.Path]:
    file_path_suffix = generate_file_path_suffix_as_id(
        tsdr_options, metric_types, time_range=time_range, use_manually_selected_metrics=use_manually_selected_metrics
    )
    dir_name: str = f"tsdr_{dataset_id}_{file_path_suffix}"
    parent_path = DATA_DIR / dir_name
    return parent_path.is_dir(), parent_path


# def load_tsdr(
#     dataset_id: str,
#     revert_normalized_time_series: bool = False,
#     suffix: str = "",
#     metric_types: dict[str, bool] = ALL_METRIC_TYPES,
# ) -> list[tuple[DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
#     ok, parent_path = check_cache_suffix(dataset_id, suffix)
#     if not ok:
#         raise FileNotFoundError(f"tsdr_{dataset_id} is not found")
#     results = []
#     for path in parent_path.iterdir():
#         with (path / "record.bz2").open("rb") as f:
#             record = joblib.load(f)
#             record.data_df = _filter_prometheus_exporter_go_metrics(
#                 filter_metrics_by_metric_type(record.data_df, metric_types)
#             )
#         with (path / "filtered_df.bz2").open("rb") as f:
#             filtered_df = _filter_prometheus_exporter_go_metrics(
#                 filter_metrics_by_metric_type(joblib.load(f), metric_types)
#             )
#         with (path / "anomalous_df.bz2").open("rb") as f:
#             anomalous_df = _filter_prometheus_exporter_go_metrics(
#                 filter_metrics_by_metric_type(joblib.load(f), metric_types)
#             )
#         with (path / "reduced_df.bz2").open("rb") as f:
#             reduced_df = _filter_prometheus_exporter_go_metrics(
#                 filter_metrics_by_metric_type(joblib.load(f), metric_types)
#             )
#             if revert_normalized_time_series:  # Workaround
#                 for metric_name, _ in reduced_df.items():
#                     reduced_df[metric_name] = anomalous_df[metric_name]
#         results.append((record, filtered_df, anomalous_df, reduced_df))
#     return results


def save_tsdr(
    dataset_id: str,
    record: DatasetRecord,
    filtered_df: pd.DataFrame,
    anomalous_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
    file_path_suffix: str,
) -> None:
    assert reduced_df.shape[1] > 0, f"{reduced_df.shape[1]} should be > 0"
    assert (
        record.data_df.shape[1] >= filtered_df.shape[1]
    ), f"{record.data_df.shape[1]} should be > {filtered_df.shape[1]}"
    assert filtered_df.shape[1] >= reduced_df.shape[1], f"{filtered_df.shape[1]} should be > {reduced_df.shape[1]}"

    dir_name: str = f"tsdr_{dataset_id}" if file_path_suffix == "" else f"tsdr_{dataset_id}_{file_path_suffix}"
    path = DATA_DIR / dir_name / record.chaos_case_full().replace("/", "_")
    path.mkdir(parents=True, exist_ok=True)
    for obj, name in (
        (record, "record"),
        (filtered_df, "filtered_df"),
        (anomalous_df, "anomalous_df"),
        (reduced_df, "reduced_df"),
    ):
        joblib.dump(obj, path / f"{name}.bz2", compress=("bz2", 3))  # type: ignore


def validate_tsdr_results(
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
