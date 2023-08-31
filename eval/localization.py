import datetime
import gc
import itertools
import logging
import os
import time
import traceback
import warnings
from enum import IntEnum
from typing import Any, Final

import joblib
import neptune
import neptune.internal.utils.logger as npt_logger
import neptune.types
import networkx as nx
import pandas as pd
from neptune.internal.hardware.gpu.gpu_monitor import GPUMonitor
from timeout_timer import TimeoutInterrupt, timeout
from tqdm.auto import tqdm

from diagnoser import diag
from eval.groundtruth import check_cause_metrics
from eval.localizaiton_score import (
    create_localization_score_as_dataframe, create_rank_as_dataframe,
    create_rank_as_dataframe_for_multiple_cases_from_frames)
from eval.tsdr import (DEFAULT_CHAOS_TYPES, METRIC_TYPES_PAIRS,
                       check_cache_suffix, load_tsdr_by_chaos)
from eval.util.logger import logger
from meltria.loader import DatasetRecord

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
GPUMonitor.nvml_error_printed = True  # Suppress NVML error messages
npt_logger.logger.setLevel(logging.ERROR)  # Suppress Neptune INFO log messages to console


class DiagTargetPhaseOption(IntEnum):
    FIRST = -2
    LAST = -1
    RAW = 0


DEFAULT_TIMEOUT_SEC: Final[int] = 3600
DEFAULT_DIAG_TARGET_PHASE_OPTION: Final[
    DiagTargetPhaseOption
] = DiagTargetPhaseOption.LAST
DEFAULT_RESULT_GROUPBY: Final[dict[str, bool]] = dict(
    group_by_cause_type=False, group_by_cause_comp=False
)


class NoResultError(Exception):
    def __init__(self, arg: str):
        self.arg = arg

    def __str__(self):
        return self.arg


def revert_true_cause_metrics(
    record: DatasetRecord, data_df: pd.DataFrame
) -> pd.DataFrame:
    _, cause_metrics = check_cause_metrics(
        pk=record.pk,
        metrics=list(record.data_df.columns),
        chaos_type=record.chaos_type(),
        chaos_comp=record.chaos_comp(),
        optional_cause=True,
    )
    true_cause_df = record.data_df.loc[:, cause_metrics.tolist()]
    not_have_true_cause_columns = true_cause_df.columns.difference(data_df.columns)
    return pd.concat([data_df, true_cause_df[not_have_true_cause_columns]], axis=1)


def diagnose_and_rank(
    dataset_id: str,
    reduced_df: pd.DataFrame,
    record: DatasetRecord,
    diag_options: dict[str, Any],
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> tuple[nx.Graph, pd.DataFrame, float] | None:
    gc.collect()  # https://github.com/joblib/joblib/issues/721#issuecomment-417216395

    if diag_options.get("enable_revert_true_cause_metrics", False):
        reduced_df = revert_true_cause_metrics(record, reduced_df)

    sta: float = time.perf_counter()

    try:
        with timeout(timeout_sec, timer="thread"):
            try:
                G, ranks = diag.build_and_walk_causal_graph(
                    reduced_df,
                    record.pk,
                    **diag_options,
                )
            except TimeoutInterrupt:
                logger.error(
                    f"TimeoutInterrupt while diagnosing {record.chaos_case_full()}, {diag_options}"
                )
                G = nx.empty_graph()
                ranks = []
    except Exception as e:
        logger.error(f"Failed to diagnose {record.chaos_case_full()}, {diag_options}: {e}\n{traceback.format_exc()}")
        return nx.empty_graph(), create_rank_as_dataframe([], dataset_id, record), 0.0

    end: float = time.perf_counter()
    elapsed: float = end - sta

    if len(ranks) == 0:
        logger.error(
            f"Failed to diagnose {record.chaos_case_full()} with {len(ranks)} ranks"
        )
        return nx.empty_graph(), create_rank_as_dataframe([], dataset_id, record), elapsed

    return G, create_rank_as_dataframe(ranks, dataset_id, record), elapsed


def diagnose_and_rank_multi_datasets(
    dataset_id: str,
    datasets: dict,
    diag_options: dict[str, float | bool | int],
    metric_types: dict[str, bool],
    diag_target_phase_option: DiagTargetPhaseOption = DEFAULT_DIAG_TARGET_PHASE_OPTION,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    n_workers: int = -1,
) -> tuple[list, dict[tuple[str, str, int], float], pd.DataFrame]:
    assert len(datasets) != 0

    records = []
    for (_, _), somethings_records in datasets.items():
        for record, data_df_by_metric_type in somethings_records:
            reduced_df = pd.concat(
                [
                    data_df_by_metric_type[metric_type][diag_target_phase_option.value]
                    for metric_type, is_enabled in metric_types.items() if is_enabled
                ],
                axis=1,
            )
            if not any(
                label in reduced_df.columns for label in record.pk.get_root_metrics()
            ):
                logger.warning(
                    f"No root metrics found in the dataset {record.chaos_case_full()}: {reduced_df.shape[1]}"
                )
                continue
            records.append((record, reduced_df))

    results = joblib.Parallel(n_jobs=n_workers)(
        joblib.delayed(diagnose_and_rank)(dataset_id, reduced_df, record, diag_options, timeout_sec)
        for record, reduced_df in records
    )
    results = [result for result in results if result is not None]
    assert results is not None
    if len(results) == 0:
        raise NoResultError(f"No result found in the dataset: {diag_options}, {metric_types}")

    elapsed_times: dict[tuple[str, str, int], float] = {
        (record.chaos_type(), record.chaos_comp(), record.chaos_case_num()): elapsed
        for (_, _, elapsed), (record, _) in zip(results, records)
    }

    return [_[1] for _ in results], elapsed_times


def calculate_mean_elapsed_time(
    elapsed_times: dict[tuple[str, str, int], float]
) -> dict:
    df = pd.DataFrame(
        [(_[0], _[1], _[2], elapsed) for _, elapsed in elapsed_times.items()],
        columns=["chaos_type", "chaos_comp", "chaos_case_num", "elapsed_time"],
    ).set_index(["chaos_type", "chaos_comp", "chaos_case_num"])
    return {
        "mean_by_chaos_type-df": neptune.types.File.as_html(
            df.groupby(level="chaos_type").mean()
        ),
        "mean_by_chaos_comp-df": neptune.types.File.as_html(
            df.groupby(level="chaos_comp").mean()
        ),
        "mean_by_chaos_type_and_chaos_comp": neptune.types.File.as_html(
            df.groupby(level=["chaos_type", "chaos_comp"]).mean()
        ),
        "mean": df.mean().to_dict(),
        "elapsed-df": neptune.types.File.as_html(df),
    }


def load_tsdr_and_localize(
    experiment_id: str,
    dataset_id: str,
    n: int,
    metric_types: dict[str, bool],
    tsdr_options: dict[str, Any],
    diag_options: dict[str, Any],
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
    target_chaos_types: set[str] = DEFAULT_CHAOS_TYPES,
    from_orig: tuple[bool, int, int, int] = (False, 0, 0, 0),  # from_orig flag, from_orig_num_datapoints, max_chaos_case_num
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    experiment_n_workers: int = -1,
) -> None:
    datasets = load_tsdr_by_chaos(
        dataset_id,
        metric_types=metric_types,
        revert_normalized_time_series=True,
        tsdr_options=tsdr_options,
        use_manually_selected_metrics=use_manually_selected_metrics,
        time_range=time_range,
        target_chaos_types=target_chaos_types,
        from_orig=from_orig,
    )

    first_record = list(datasets.values())[0][0][0]
    pk = first_record.pk

    run = neptune.init_run(
        project=os.environ["TSDR_LOCALIZATION_NEPTUNE_PROJECT"],
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )
    run["experiment_id"] = experiment_id
    run["experiment_n_workers"] = experiment_n_workers
    run["dataset"] = {
        "dataset_id": dataset_id,
        "target_app": first_record.target_app(),
        "metric_types": metric_types,
        "use_manually_selected_metrics": use_manually_selected_metrics,
    }
    run["parameters/tsdr"] = tsdr_options
    run["parameters"] = diag_options

    try:
        data_dfs_by_metric_type, elapsed_times = diagnose_and_rank_multi_datasets(
            dataset_id,
            datasets,
            diag_options,
            metric_types,
            timeout_sec=timeout_sec,
            n_workers=experiment_n_workers,
        )
    except NoResultError as e:
        logger.error(e)
        run.stop()
        return

    run["elapsed_time"] = calculate_mean_elapsed_time(elapsed_times)

    df = create_localization_score_as_dataframe(
        data_dfs_by_metric_type, pk=pk, metric_types=metric_types, k=n,
    )
    run["scores/metric/num_cases"] = df.at[1, "#cases (metric)"]
    run["scores/container/num_cases"] = df.at[1, "#cases (container)"]
    run["scores/service/num_cases"] = df.at[1, "#cases (service)"]
    for ind, row in df.iterrows():
        for name in ["AC", "AVG"]:
            run[f"scores/metric/{name}_{ind}"] = row[f"{name}@K (metric)"]
            run[f"scores/container/{name}_{ind}"] = row[f"{name}@K (container)"]
            run[f"scores/service/{name}_{ind}"] = row[f"{name}@K (service)"]
            run[f"scores/metric/{name}_{ind}_mand"] = row[
                f"{name}@K (metric) (mandatory)"
            ]

    run["eval/ranks-df"] = neptune.types.File.as_html(
        create_rank_as_dataframe_for_multiple_cases_from_frames(data_dfs_by_metric_type, n=30),
    )
    run["eval/score-df"] = neptune.types.File.as_html(df)
    run["eval/score-df-by-cause-comp"] = neptune.types.File.as_html(
        create_localization_score_as_dataframe(
            data_dfs_by_metric_type,
            pk=pk,
            k=n,
            group_by_cause_comp=True,
            group_by_cause_type=False,
            metric_types=metric_types,
        )
    )
    run["eval/score-df-by-cause-type"] = neptune.types.File.as_html(
        create_localization_score_as_dataframe(
            data_dfs_by_metric_type,
            pk=pk,
            k=n,
            group_by_cause_comp=False,
            group_by_cause_type=True,
            metric_types=metric_types,
        )
    )
    run["eval/score-df-by-cause-comp-and-type"] = neptune.types.File.as_html(
        create_localization_score_as_dataframe(
            data_dfs_by_metric_type,
            pk=pk,
            k=n,
            group_by_cause_comp=True,
            group_by_cause_type=True,
            metric_types=metric_types,
        )
    )
    run.stop()


def sweep_localization(
    dataset_id: str,
    n: int,
    list_of_tsdr_options: list[dict[str, Any]],
    list_of_diag_options: list[dict[str, Any]],
    pair_of_use_manually_selected_metrics: list[bool],
    metric_types_pairs: list[dict[str, bool]] = METRIC_TYPES_PAIRS,
    time_ranges: list[tuple[int, int]] = [(0, 0)],
    target_chaos_types: set[str] = DEFAULT_CHAOS_TYPES,
    from_orig: tuple[bool, int, int, int] = (False, 0, 0, 0),  # from_orig flag, from_orig_num_datapoints, max_chaos_case_num
    experiment_id: str = "",
    experiment_n_workers: int = -1,
    progress: bool = False,
    resuming_no: int = 0,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> None:
    if experiment_id == "":
        experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    for (
        tsdr_options,
        metric_types,
        use_manually_selected_metrics,
        time_range,
    ) in itertools.product(
        list_of_tsdr_options,
        metric_types_pairs,
        pair_of_use_manually_selected_metrics,
        time_ranges,
    ):
        if metric_types["middlewares"] and use_manually_selected_metrics:
            continue
        if not check_cache_suffix(
            dataset_id,
            metric_types,
            tsdr_options,
            use_manually_selected_metrics,
            time_range,
        ):
            raise ValueError(
                f"None of cache is available for {metric_types}, use_manually_selected_metric={use_manually_selected_metrics}, {tsdr_options}"
            )

    params = [
        items for items in itertools.product(
            list_of_diag_options,
            list_of_tsdr_options,
            metric_types_pairs,
            pair_of_use_manually_selected_metrics,
            time_ranges,
        )
        if not (items[2]["middlewares"] and items[3])  # skip middlewares with use_manually_selected_metrics=True
    ]
    if progress:
        params = tqdm(params, desc="sweeping localization", dynamic_ncols=True)
    for i, (
        diag_options,
        tsdr_options,
        metric_types,
        use_manually_selected_metrics,
        time_range,
    ) in enumerate(params, 1):
        if resuming_no > i:
            continue
        tqdm.write(
            f"{i}/{len(params)}: Starting experiment {experiment_id} with {metric_types}, {diag_options} and {tsdr_options}"
        )
        load_tsdr_and_localize(
            experiment_id=experiment_id,
            experiment_n_workers=experiment_n_workers,
            dataset_id=dataset_id,
            n=n,
            metric_types=metric_types,
            tsdr_options=tsdr_options,
            diag_options=diag_options,
            use_manually_selected_metrics=use_manually_selected_metrics,
            time_range=time_range,
            target_chaos_types=target_chaos_types,
            from_orig=from_orig,
            timeout_sec=timeout_sec,
        )
