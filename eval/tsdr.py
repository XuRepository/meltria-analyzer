import datetime
import hashlib
import itertools
import logging
import os
import pathlib
from collections import defaultdict
from io import StringIO
from multiprocessing import cpu_count
from typing import Any, Final

import joblib
import neptune
import neptune.internal.utils.logger as npt_logger
import neptune.types
import numpy as np
import numpy.typing as npt
import pandas as pd
from neptune.internal.hardware.gpu.gpu_monitor import GPUMonitor
from tqdm.auto import tqdm

from eval import validation
from eval.groundtruth import check_cause_metrics
from eval.util.logger import logger
from meltria import loader
from meltria.metric_types import ALL_METRIC_TYPES, METRIC_PREFIX_TO_TYPE
from tsdr import tsdr

RAW_DATA_DIR = "/datasets"
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

GPUMonitor.nvml_error_printed = True  # Suppress NVML error messages
npt_logger.logger.setLevel(logging.ERROR)  # Suppress Neptune INFO log messages to console


def generate_file_path_suffix_as_id(
    tsdr_options: dict[str, Any],
    metric_types: dict[str, bool],
    time_range: tuple[int, int],
    **kwargs: Any,
) -> str:
    return hashlib.md5(
        "".join(
            [f"{k}{v}" for k, v in sorted(tsdr_options.items())]
            + [f"{k}{v}" for k, v in sorted(metric_types.items())]
            + (
                []
                if time_range[0] == 0 and time_range[1] == 0
                else [f"time_range{time_range[0]}{time_range[1]}"]
            )
            + [f"{k}{v}" for k, v in sorted(kwargs.items())]
        ).encode()
    ).hexdigest()


def sweep_tsdr_and_save_as_cache(
    dataset_id: str,
    records: list[loader.DatasetRecord],
    list_of_tsdr_options: list[dict[str, Any]],
    use_manually_selected_metrics: list[bool] = [True, False],
    metric_types_pairs: list[dict[str, bool]] = METRIC_TYPES_PAIRS,
    time_ranges: list[tuple[int, int]] = [(0, 0)],
    experiment_id: str = "",
    progress: bool = False,
    resuming_no: int = 0,
) -> None:
    experiment_id = experiment_id or datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    combinations = [
        items for items in itertools.product(
            list_of_tsdr_options,
            metric_types_pairs,
            use_manually_selected_metrics,
            time_ranges,
        ) if not (items[1]["middlewares"] and items[2])  # skip middlewares with use_manually_selected_metrics=True
    ]
    if progress:
        combinations = tqdm(combinations, desc="sweeping tsdr", dynamic_ncols=True)

    for i, (tsdr_options, metric_types, _use_manually_selected_metrics, time_range) in enumerate(combinations, 1):
        if resuming_no > i:
            continue
        tqdm.write(
            f"{i}/{len(combinations)}: Starting experiment {experiment_id} with {metric_types}, {tsdr_options}, {time_range}, use_manually_selected_metrics={_use_manually_selected_metrics}"
        )
        run_tsdr_and_save_as_cache_with_tracking(
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            metric_types=metric_types,
            records=records,
            tsdr_options=tsdr_options,
            time_range=time_range,
            use_manually_selected_metrics=_use_manually_selected_metrics,
        )


def run_tsdr_and_save_as_cache_with_tracking(
    experiment_id: str,
    dataset_id: str,
    metric_types: dict[str, bool],
    records: list[loader.DatasetRecord],
    tsdr_options: dict[str, Any] = tsdr_default_options,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
    data_dir: pathlib.Path = DATA_DIR,
) -> None:
    run = neptune.init_run(
        project=os.environ["TSDR_NEPTUNE_PROJECT"],
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )
    run["experiment_id"] = experiment_id
    run["dataset/dataset_id"] = dataset_id
    run["dataset/target_app"] = records[0].target_app()
    run["dataset/metric_types"] = metric_types
    run["dataset/use_manually_selected_metrics"] = use_manually_selected_metrics
    run["dataset/time_range/start"] = time_range[0]
    run["dataset/time_range/end"] = time_range[1]
    run["parameters"] = tsdr_options

    scores_df, clusters_stats = run_tsdr_and_save_as_cache(
        dataset_id=dataset_id,
        metric_types=metric_types,
        records=records,
        tsdr_options=tsdr_options,
        use_manually_selected_metrics=use_manually_selected_metrics,
        time_range=time_range,
        data_dir=data_dir,
    )

    upload_scores_to_neptune(run, scores_df, metric_types)
    upload_clusters_to_neptune(run, clusters_stats)
    run.stop()


def run_tsdr_and_save_as_cache(
    dataset_id: str,
    metric_types: dict[str, bool],
    records: list[loader.DatasetRecord],
    tsdr_options: dict[str, Any] = tsdr_default_options,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
    data_dir: pathlib.Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Workaround
    # if (
    #     metric_types["middlewares"]
    #     and tsdr_options.get("step2_dbscan_algorithm") == "dbscan"
    #     and tsdr_options.get("step2_dbscan_dist_type") == "pearsonr"
    # ):
    #     logger.info("Skip dbscan with pearsonr to dataset including middlewares because it takes too long time.")
    #     return

    file_path_suffix = generate_file_path_suffix_as_id(
        tsdr_options,
        metric_types,
        time_range=time_range,
        use_manually_selected_metrics=use_manually_selected_metrics,
    )

    score_dfs: list[pd.DataFrame] = []
    clusters_stats: list[pd.DataFrame] = []
    for record in records:
        (
            prefiltered_df,
            filtered_df,
            anomalous_df,
            reduced_df,
            tsdr_stat,
            clusters_stat,
        ) = run_tsdr(
            record,
            tsdr_options=tsdr_options,
            enable_unireducer=tsdr_options["enable_unireducer"],
            enable_multireducer=tsdr_options["enable_multireducer"],
            metric_types=metric_types,
            use_manually_selected_metrics=use_manually_selected_metrics,
            time_range=time_range,
        )
        score_dfs.append(
            calculate_scores_from_tsdr_result(tsdr_stat, record, metric_types)
        )

        clusters_stat["dataset_id"] = dataset_id
        clusters_stat["chaos_type"] = record.chaos_type()
        clusters_stat["chaos_comp"] = record.chaos_comp()
        clusters_stat["chaos_case_num"] = record.chaos_case_num()
        clusters_stats.append(clusters_stat)

        # TODO: should save prefiltered_df?
        save_tsdr(
            dataset_id,
            record,
            filtered_df,
            anomalous_df,
            reduced_df,
            file_path_suffix=file_path_suffix,
            data_dir=data_dir,
        )
        del (
            record,
            prefiltered_df,
            filtered_df,
            anomalous_df,
            reduced_df,
        )  # for memory efficiency

    return pd.concat(score_dfs, axis=0), pd.concat(clusters_stats, axis=0)


def run_tsdr(
    record: loader.DatasetRecord,
    tsdr_options: dict[str, Any],
    enable_unireducer: bool = True,
    enable_multireducer: bool = True,
    metric_types: dict[str, bool] = ALL_METRIC_TYPES,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
    max_workers: int = -1,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[tuple[pd.DataFrame, pd.DataFrame, float]],
    pd.DataFrame,
]:
    tsdr_options = dict(tsdr_default_options, **tsdr_options)

    prefiltered_df = record.data_df
    if use_manually_selected_metrics:
        prefiltered_df = filter_manually_selected_metrics(prefiltered_df)

    start, end = time_range
    if end == 0:
        end = prefiltered_df.shape[0]
    prefiltered_df = filter_metrics_by_metric_type(
        prefiltered_df.iloc[start:end, :], metric_types
    )

    reducer = tsdr.Tsdr(tsdr_options["step1_method_name"], **tsdr_options)
    tsdr_stat, clusters_stat, _ = reducer.run(
        X=prefiltered_df,
        pk=record.pk,
        max_workers=cpu_count() if max_workers == -1 else max_workers,
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
    return (
        prefiltered_df,
        filtered_df,
        anomalous_df,
        reduced_df,
        tsdr_stat,
        clusters_stat,
    )


def run_tsdr_and_save_as_cache_from_orig_data(
    dataset_id: str,
    tsdr_options: dict[str, Any],
    num_datapoints: int,
    metric_types: dict[str, bool] = ALL_METRIC_TYPES,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
    max_chaos_case_num: int = 0,
    num_faulty_datapoints: int = 0,
    data_dir: pathlib.Path = DATA_DIR,
    target_chaos_types: set[str] = DEFAULT_CHAOS_TYPES,
) -> None:
    metrics_files = loader.find_metrics_files(RAW_DATA_DIR, dataset_id=dataset_id)
    records = loader.load_dataset(
        metrics_files,
        target_metric_types=metric_types,
        num_datapoints=num_datapoints,
        max_chaos_case_num=max_chaos_case_num,
        validated=True,
        num_faulty_datapoints=num_faulty_datapoints,
        target_chaos_types=target_chaos_types,
    )
    run_tsdr_and_save_as_cache(
        records=records,
        dataset_id=dataset_id,
        metric_types=metric_types,
        tsdr_options=tsdr_options,
        use_manually_selected_metrics=use_manually_selected_metrics,
        time_range=time_range,
        data_dir=data_dir,
    )


def sweep_tsdr_for_recall(
    records: list[loader.DatasetRecord],
    list_of_tsdr_options: list[dict[str, Any]],
    use_manually_selected_metrics: list[bool] = [True, False],
    metric_types_pairs: list[dict[str, bool]] = METRIC_TYPES_PAIRS,
    time_range: tuple[int, int] = (0, 0),
) -> pd.DataFrame:
    score_dfs: list[pd.DataFrame] = []
    for tsdr_options in list_of_tsdr_options:
        for metric_types in metric_types_pairs:
            for use_manually_selected_metric in use_manually_selected_metrics:
                score_df = run_tsdr_for_recall(
                    metric_types=metric_types,
                    records=records,
                    tsdr_options=tsdr_options,
                    time_range=time_range,
                    use_manually_selected_metrics=use_manually_selected_metric,
                )
                for opt, val in tsdr_options.items():
                    score_df[opt] = val
                score_df["use_manually_selected_metrics"] = use_manually_selected_metric
                for m, ok in metric_types.items():
                    score_df[f"metric_types/{m}"] = ok
                score_df["time_range/start"] = time_range[0]
                score_df["time_range/end"] = time_range[1]
                score_dfs.append(score_df)
    return pd.concat(score_dfs, axis=0)


def transform_record_only_metrics_in_cause_services(record: loader.DatasetRecord) -> loader.DatasetRecord:
    sli_df = record.data_df[
        [m for m in record.pk.get_root_metrics() if m in record.data_df.columns]
    ]
    ctnr = record.chaos_comp()
    service = record.pk.get_service_by_container(ctnr)
    cause_service_df = record.data_df.loc[
        :,
        record.data_df.columns.str.startswith(
            (f"s-{service}_", f"c-{ctnr}_", f"m-{ctnr}_")
        ),
    ]
    record.data_df = pd.concat([sli_df, cause_service_df], axis=1)
    return record


def run_tsdr_for_recall(
    metric_types: dict[str, bool],
    records: list[loader.DatasetRecord],
    tsdr_options: dict[str, Any] = tsdr_default_options,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
) -> pd.DataFrame:
    records = [
        transform_record_only_metrics_in_cause_services(record) for record in records
    ]

    tsdr_results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(run_tsdr)(
            record=record,
            tsdr_options=tsdr_options,
            enable_unireducer=tsdr_options["enable_unireducer"],
            enable_multireducer=tsdr_options["enable_multireducer"],
            metric_types=metric_types,
            use_manually_selected_metrics=use_manually_selected_metrics,
            time_range=time_range,
            max_workers=1,
        )
        for record in records
    )
    assert tsdr_results is not None

    scores: list[dict[str, float]] = []
    for record, (prefiltered_df, filtered_df, anomalous_df, reduced_df, _, _) in zip(
        records, tsdr_results
    ):
        total_mandatory_cause_metrics = check_cause_metrics(
            record.pk,
            prefiltered_df.columns.tolist(),
            record.chaos_type(),
            record.chaos_comp(),
            optional_cause=False,
        )[1]
        total_cause_metrics = check_cause_metrics(
            record.pk,
            prefiltered_df.columns.tolist(),
            record.chaos_type(),
            record.chaos_comp(),
            optional_cause=True,
        )[1]
        score: dict[str, Any] = {}
        for name, df in zip(
            ["filtered", "anomalous", "reduced"],
            [filtered_df, anomalous_df, reduced_df],
        ):
            _, found_mandatory_metrics = check_cause_metrics(
                record.pk,
                df.columns.tolist(),
                record.chaos_type(),
                record.chaos_comp(),
                optional_cause=False,
            )
            _, found_metrics = check_cause_metrics(
                record.pk,
                df.columns.tolist(),
                record.chaos_type(),
                record.chaos_comp(),
                optional_cause=True,
            )
            score[f"{name}/mand_recall"] = recall_of_cause_metrics(
                set(total_mandatory_cause_metrics.tolist()),
                set(found_mandatory_metrics.tolist()),
            )
            score[f"{name}/n_total_mand_metrics"] = len(total_mandatory_cause_metrics)
            score[f"{name}/n_found_mand_metrics"] = len(found_mandatory_metrics)
            score[f"{name}/recall"] = recall_of_cause_metrics(
                set(total_cause_metrics), set(found_metrics.tolist())
            )
            score[f"{name}/n_total_metrics"] = len(total_cause_metrics)
            score[f"{name}/n_found_metrics"] = len(found_metrics)
        score.update(
            {
                "chaos_type": record.chaos_type(),
                "chaos_comp": record.chaos_comp(),
                "chaos_case_num": record.chaos_case_num(),
            }
        )
        scores.append(score)
        del record, filtered_df, anomalous_df, reduced_df  # for memory efficiency

    return pd.DataFrame(scores)


def _get_cause_metrics(
    record: loader.DatasetRecord, metrics: list, optional_cause: bool = True
) -> list[str]:
    cause_metrics_exist, found_metrics = check_cause_metrics(
        pk=record.pk,
        metrics=metrics,
        chaos_type=record.chaos_type(),
        chaos_comp=record.chaos_comp(),
        optional_cause=optional_cause,
    )
    if not cause_metrics_exist:
        logger.warning(
            f"Cause metrics not found: pk={record.pk}, chaos_type={record.chaos_type()}, chaos_comp={record.chaos_comp()}"
        )
    return found_metrics.tolist()


def recall_of_cause_metrics(
    total_cause_metrics: set[str], found_cause_metrics: set[str]
) -> float:
    return len(total_cause_metrics & found_cause_metrics) / len(total_cause_metrics)


def proportion_of_cause_metrics(
    total_metrics: set[str], found_cause_metrics: set[str]
) -> float:
    return len(found_cause_metrics) / len(total_metrics)


def calculate_scores_from_tsdr_result(
    tsdr_stat: list[tuple[pd.DataFrame, pd.DataFrame, float]],
    record: loader.DatasetRecord,
    metric_types: dict[str, bool],
) -> pd.DataFrame:
    # metrics denominator after phase0 simple filtering
    total_cause_metrics: set[str] = set(
        _get_cause_metrics(record, list(tsdr_stat[1][0].columns))
    )
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
        _, found_mandatory_metrics = check_cause_metrics(
            pk=record.pk,
            metrics=list(reduced_df.columns),
            chaos_type=record.chaos_type(),
            chaos_comp=record.chaos_comp(),
            optional_cause=False,
        )
        num_series_by_type: dict[str, int] = {}
        for metric_type, enable in metric_types.items():
            if not enable:
                continue
            num_series_by_type[f"num_series/{metric_type}/raw"] = (
                tsdr_stat[0][1].loc[metric_type]["count"].sum()
            )
            num_series_by_type[f"num_series/{metric_type}/filtered"] = (
                tsdr_stat[1][1].loc[metric_type]["count"].sum()
                if metric_type in tsdr_stat[1][1]
                else 0
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
                    total_mandatory_cause_metrics, set(found_mandatory_metrics)
                ),
                "cause_metrics/recall": recall_of_cause_metrics(
                    total_cause_metrics, set(found_metrics)
                ),
                "cause_metrics/proportion": proportion_of_cause_metrics(
                    set(reduced_df.columns), set(found_metrics)
                ),
                "cause_metrics/only_mandatory_proportion": proportion_of_cause_metrics(
                    set(reduced_df.columns), set(found_mandatory_metrics)
                ),
                "cause_metrics/num_total": len(total_cause_metrics),
                "cause_metrics/num_mandatory_total": len(total_mandatory_cause_metrics),
                "cause_metrics/num_found": len(found_metrics),
                "cause_metrics/num_mandatory_found": len(found_mandatory_metrics),
                "num_series/total/raw": tsdr_stat[0][1]["count"].sum(),  # raw
                "num_series/total/filtered": tsdr_stat[1][1][
                    "count"
                ].sum(),  # after step0
                "num_series/total/reduced": stat_df["count"].sum(),  # after step{i}
                **num_series_by_type,
                "elapsed_time": elapsed_time,
                "found_metrics": ",".join(found_metrics),
            }
        )
    return pd.DataFrame(tests).set_index(
        ["chaos_type", "chaos_comp", "metrics_file", "phase"]
    )


def get_scores_of_random_selection(num_metrics: npt.ArrayLike, num_found_metrics: npt.ArrayLike, max_k: int = 5):
    def ac_k(n, g, k: int):
        prob_single_correct = g / n
        prob_at_least_one_correct = 1 - (1 - prob_single_correct) ** k
        return prob_at_least_one_correct.mean()

    def avg_k(ac_k_) -> dict:
        return {k: sum([ac_k_[j] for j in range(1, k + 1)]) / k for k in range(1, max_k + 1)}

    ac_k_ = {k: ac_k(num_metrics, num_found_metrics, k) for k in range(1, max_k + 1)}
    avg_k_ = avg_k(ac_k_)

    return dict({f"AC_{k}": v for k, v in ac_k_.items()}, **{f"AVG_{k}": v for k, v in avg_k_.items()})


def _upload_dataframe_to_neptune(run: neptune.Run, df: pd.DataFrame, prefix: str) -> None:
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    run[f"{prefix}-csv"].upload(neptune.types.File.from_stream(csv_buffer, extension="csv"))
    run[f"{prefix}-html"].upload(neptune.types.File.as_html(df))


def upload_scores_to_neptune(
    run: neptune.Run, tests_df: pd.DataFrame, target_metric_types: dict[str, bool]
) -> None:
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
            "cause_metrics/recall_mandatory_mean": x[
                "cause_metrics/only_mandatory_recall"
            ].mean(),
            "cause_metrics/proportion_mean": x["cause_metrics/proportion"].mean(),
            "cause_metrics/proportion_mandatory_mean": x[
                "cause_metrics/only_mandatory_proportion"
            ].mean(),
            "cause_metrics/num_total_mean": x["cause_metrics/num_total"].mean(),
            "cause_metrics/num_mandatory_total_mean": x["cause_metrics/num_mandatory_total"].mean(),
            "cause_metrics/num_found_mean": x["cause_metrics/num_found"].mean(),
            "cause_metrics/num_mandatory_found_mean": x["cause_metrics/num_mandatory_found"].mean(),
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
            "random_selection_perf": get_scores_of_random_selection(
                x["num_series/total/reduced"],
                x["cause_metrics/num_mandatory_found"],
            ),
        }
        return pd.Series(d)

    scores_by_phase = (
        tests_df.groupby("phase").apply(agg_score).reset_index().set_index("phase")
    )
    scores_by_chaos_type = (
        tests_df.groupby(["chaos_type", "phase"])
        .apply(agg_score)
        .reset_index()
        .set_index(["chaos_type", "phase"])
    )
    scores_by_chaos_comp = (
        tests_df.groupby(["chaos_comp", "phase"])
        .apply(agg_score)
        .reset_index()
        .set_index(["chaos_comp", "phase"])
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

    for _df, prefix in zip(
        [tests_df, scores_by_phase, scores_by_chaos_type, scores_by_chaos_comp, scores_by_chaos_type_and_comp],
        [
            "scores/summary",
            "scores/summary_by_phase",
            "scores/summary_by_chaos_type",
            "scores/summary_by_chaos_comp",
            "scores/summary_by_chaos_type_and_chaos_comp",
        ],
    ):
        _upload_dataframe_to_neptune(run, _df, prefix)


def upload_clusters_to_neptune(run: neptune.Run, clusters_stats: pd.DataFrame) -> None:
    if clusters_stats.empty:
        return

    clusters_stats.set_index(
        ["dataset_id", "chaos_type", "chaos_comp", "chaos_case_num"], inplace=True
    )
    run["clusters_stats/details"].upload(neptune.types.File.as_html(clusters_stats))

    num_clusters = clusters_stats.groupby(
        ["dataset_id", "chaos_type", "chaos_comp", "chaos_case_num", "component"]
    ).size()
    run["clusters_stats/num_clusters-df"].upload(
        neptune.types.File.as_html(num_clusters.to_frame())
    )
    run["clusters_stats/num_clusters"] = num_clusters.agg(
        ["mean", "max", "min", "std"]
    ).to_dict()

    cluster_size = clusters_stats.groupby(
        [
            "dataset_id",
            "chaos_type",
            "chaos_comp",
            "chaos_case_num",
            "component",
            "rep_metric",
        ]
    )["sub_metrics"].apply(lambda x: np.array(x[0]).flatten().size + 1)
    run["clusters_stats/cluster_size-df"].upload(
        neptune.types.File.as_html(cluster_size.to_frame())
    )
    run["clusters_stats/cluster_size"] = cluster_size.agg(
        ["mean", "max", "min", "std"]
    ).to_dict()


def filter_metrics_by_metric_type(
    df: pd.DataFrame, metric_types: dict[str, bool]
) -> pd.DataFrame:
    return df[
        [
            metric_name
            for metric_name in df.columns.tolist()
            for metric_type, is_selected in metric_types.items()
            if is_selected
            and metric_name.startswith(METRIC_PREFIX_TO_TYPE[metric_type])
        ]
    ]


MANUALLY_SELECTED_METRICS: Final[set[str]] = {
    # "file_descriptors",
    # "processes",
    # "sockets",
    # "threads",
    # "cpu_system_seconds_total",
    "cpu_usage_seconds_total",
    # "cpu_user_seconds_total",
    # "memory_rss",
    # "memory_usage_bytes",
    "memory_working_set_bytes",
    # "fs_reads_bytes_total",
    "fs_reads_total",
    # "fs_writes_bytes_total",
    "fs_writes_total",
    "network_receive_bytes_total",
    # "network_receive_packets_total",
    "network_transmit_bytes_total",
    # "network_transmit_packets_total",
}


def filter_manually_selected_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Filter only container metrics
    return df.loc[
        :,
        [
            metric_name.startswith("s-")
            or metric_name.startswith("n-")
            or any(
                [
                    metric_name.endswith(base_name)
                    for base_name in MANUALLY_SELECTED_METRICS
                ]
            )
            for metric_name in df.columns.tolist()
        ],
    ]


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
    validation_filtering: tuple[bool, int] = (False, 0),
    from_orig: tuple[bool, int, int, int] = (False, 0, 0, 0),  # from_orig flag, from_orig_num_datapoints, max_chaos_case_num
    n_jobs: int = -1,
) -> dict[
    tuple[str, str],
    list[
        tuple[loader.DatasetRecord, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]
    ],
]:  # (chaos_type, chaos_comp)
    datasets = load_tsdr_grouped_by_metric_type(
        dataset_id,
        metric_types,
        revert_normalized_time_series,
        tsdr_options,
        use_manually_selected_metrics,
        time_range,
        validation_filtering,
        from_orig,
        target_chaos_types,
        n_jobs,
    )
    results = defaultdict(list)
    for record, df_by_metric_type in datasets:
        if record.chaos_type() not in target_chaos_types:
            continue
        results[(record.chaos_type(), record.chaos_comp())].append(
            (record, df_by_metric_type)
        )
    return results


def load_tsdr_grouped_by_metric_type(
    dataset_id: str,
    metric_types: dict[str, Any],
    revert_normalized_time_series: bool = False,
    tsdr_options: dict[str, Any] = tsdr_default_options,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
    validation_filtering: tuple[bool, int] = (False, 0),
    from_orig: tuple[bool, int, int, int] = (False, 0, 0, 0),  # from_orig flag, from_orig_num_datapoints, max_chaos_case_num
    target_chaos_types: set[str] = DEFAULT_CHAOS_TYPES,
    n_jobs: int = -1,
) -> list[
    tuple[loader.DatasetRecord, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]
]:
    from_orig_ok, from_orig_num_datapoints, from_orig_num_faulty_datapoints, max_chaos_case_num = from_orig
    if from_orig_ok:
        run_tsdr_and_save_as_cache_from_orig_data(
            dataset_id=dataset_id,
            tsdr_options=tsdr_options,
            num_datapoints=from_orig_num_datapoints,
            metric_types=metric_types,
            use_manually_selected_metrics=use_manually_selected_metrics,
            time_range=time_range,
            num_faulty_datapoints=from_orig_num_faulty_datapoints,
            target_chaos_types=target_chaos_types,
            max_chaos_case_num=max_chaos_case_num,
        )

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

    def load_data(
        path: pathlib.Path,
    ) -> (
        tuple[loader.DatasetRecord, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]
        | None
    ):
        with (path / "record.bz2").open("rb") as f:
            record = joblib.load(f)
            valid, faulty_pos = validation_filtering
            if valid:
                sli_ok = validation.find_records_detected_anomalies_of_sli(
                    [record], faulty_pos
                )
                cause_ok = validation.find_records_detected_anomalies_of_cause_metrics(
                    [record], faulty_pos
                )
                if not (sli_ok and cause_ok):
                    return None
            record.data_df = record.data_df
        with (path / "filtered_df.bz2").open("rb") as f:
            filtered_df = joblib.load(f)
            filtered_df = _group_by_metric_type(filtered_df)
        with (path / "anomalous_df.bz2").open("rb") as f:
            anomalous_df = joblib.load(f)
            anomalous_df = _group_by_metric_type(anomalous_df)
        with (path / "reduced_df.bz2").open("rb") as f:
            reduced_df = joblib.load(f)
            reduced_df = _group_by_metric_type(reduced_df)
        df_by_metric_type: dict[
            str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ] = {}
        for metric_type, is_enabled in metric_types.items():
            if not is_enabled:
                continue
            if revert_normalized_time_series:  # Workaround
                for metric_name, _ in reduced_df[metric_type].items():
                    reduced_df[metric_type][metric_name] = anomalous_df[metric_type][
                        metric_name
                    ]
            df_by_metric_type[metric_type] = (
                filtered_df[metric_type],
                anomalous_df[metric_type],
                reduced_df[metric_type],
            )
        return record, df_by_metric_type

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(load_data)(path)
        for path in parent_path.iterdir()
        if any(path.iterdir())
    )
    assert results is not None
    return [result for result in results if result is not None]


def check_cache_suffix(
    dataset_id: str,
    metric_types: dict[str, Any],
    tsdr_options: dict[str, Any] = tsdr_default_options,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
) -> tuple[bool, pathlib.Path]:
    file_path_suffix = generate_file_path_suffix_as_id(
        tsdr_options,
        metric_types,
        time_range=time_range,
        use_manually_selected_metrics=use_manually_selected_metrics,
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
    record: loader.DatasetRecord,
    filtered_df: pd.DataFrame,
    anomalous_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
    file_path_suffix: str,
    data_dir: pathlib.Path = DATA_DIR,
) -> None:
    assert reduced_df.shape[1] > 0, f"{reduced_df.shape[1]} should be > 0"
    assert (
        record.data_df.shape[1] >= filtered_df.shape[1]
    ), f"{record.data_df.shape[1]} should be > {filtered_df.shape[1]}"
    assert (
        filtered_df.shape[1] >= reduced_df.shape[1]
    ), f"{filtered_df.shape[1]} should be > {reduced_df.shape[1]}"

    dir_name: str = (
        f"tsdr_{dataset_id}"
        if file_path_suffix == ""
        else f"tsdr_{dataset_id}_{file_path_suffix}"
    )
    path = data_dir / dir_name / record.chaos_case_full().replace("/", "_")
    path.mkdir(parents=True, exist_ok=True)
    for obj, name in (
        (record, "record"),
        (filtered_df, "filtered_df"),
        (anomalous_df, "anomalous_df"),
        (reduced_df, "reduced_df"),
    ):
        joblib.dump(obj, path / f"{name}.bz2", compress=("bz2", 3))  # type: ignore


def validate_tsdr_results(
    datasets: list[tuple[loader.DatasetRecord, pd.DataFrame, pd.DataFrame, pd.DataFrame]]
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
        dataset_by_chaos_case[
            (record.chaos_type(), record.chaos_comp(), record.chaos_case_num())
        ] = (
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
