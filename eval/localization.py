import datetime
import itertools
import logging
import os
import time
import warnings
from collections import defaultdict
from enum import IntEnum
from typing import Any, Final

import joblib
import neptune
import neptune.types
import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

from diagnoser import diag
from eval.groundtruth import check_cause_metrics
from eval.localizaiton_score import (
    create_localization_score_as_dataframe, create_rank_as_dataframe,
    create_rank_as_dataframe_for_multiple_cases_from_frames)
from eval.tsdr import (DEFAULT_CHAOS_TYPES, METRIC_TYPES_PAIRS,
                       check_cache_suffix, load_tsdr_by_chaos)
from meltria.loader import DatasetRecord

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


class DiagTargetPhaseOption(IntEnum):
    FIRST = -2
    LAST = -1
    RAW = 0


DEFAULT_DIAG_TARGET_PHASE_OPTION: Final[
    DiagTargetPhaseOption
] = DiagTargetPhaseOption.LAST
DEFAULT_RESULT_GROUPBY: Final[dict[str, bool]] = dict(
    group_by_cause_type=False, group_by_cause_comp=False
)

DIAG_DEFAULT_OPTIONS: Final[dict[str, str | float | bool]] = dict(
    enable_prior_knowledge=False,
    pc_library="cdt",
    cg_algo="pc",
    pc_citest_alpha=0.05,
    pc_citest="gaussian",
    pc_variant="stable",
    disable_orientation=False,
    disable_ci_edge_cut=False,
    walk_method="monitorrank",
    root_metric_type="latency",
)


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
    diag_options: dict[str, str | float | bool] = DIAG_DEFAULT_OPTIONS,
) -> tuple[nx.Graph, pd.DataFrame, float] | None:
    if diag_options.get("enable_revert_true_cause_metrics", False):
        reduced_df = revert_true_cause_metrics(record, reduced_df)

    sta: float = time.perf_counter()

    opts = dict(DIAG_DEFAULT_OPTIONS, **diag_options)
    G, ranks = diag.build_and_walk_causal_graph(
        reduced_df,
        record.pk,
        **opts,
    )

    end: float = time.perf_counter()
    elapsed: float = end - sta

    if len(ranks) == 0:
        logging.error(
            f"Failed to diagnose {record.chaos_case_full()} with {len(ranks)} ranks"
        )
        return None
    return G, create_rank_as_dataframe(ranks, dataset_id, record), elapsed


def diagnose_and_rank_multi_datasets(
    dataset_id: str,
    datasets: dict,
    diag_options: dict[str, float | bool | int],
    diag_target_phase_option: DiagTargetPhaseOption = DEFAULT_DIAG_TARGET_PHASE_OPTION,
    n_workers: int = -1,
) -> tuple[list, dict[tuple[str, str, int], float]]:
    assert len(datasets) != 0

    records = []
    for (_, _), somethings_records in datasets.items():
        for record, data_df_by_metric_type in somethings_records:
            reduced_df = pd.concat(
                [
                    data_df_by_metric_type["services"][diag_target_phase_option.value],
                    data_df_by_metric_type["containers"][
                        diag_target_phase_option.value
                    ],
                    data_df_by_metric_type["middlewares"][
                        diag_target_phase_option.value
                    ],
                    data_df_by_metric_type["nodes"][diag_target_phase_option],
                ],
                axis=1,
            )
            if not any(
                label in reduced_df.columns for label in record.pk.get_root_metrics()
            ):
                logging.warn(
                    f"No root metrics found in the dataset {record.chaos_case_full()}: {reduced_df.shape[1]}"
                )
                continue
            records.append((record, reduced_df))

    results = joblib.Parallel(n_jobs=n_workers)(
        joblib.delayed(diagnose_and_rank)(dataset_id, reduced_df, record, diag_options)
        for record, reduced_df in records
    )
    results = [result for result in results if result is not None]
    assert results is not None
    assert len(results) != 0

    elapsed_times: dict[tuple[str, str, int], float] = {
        (record.chaos_type(), record.chaos_comp(), record.chaos_case_num()): elapsed
        for (_, _, elapsed), (record, _) in zip(results, records)
    }

    return [_[1] for _ in results], elapsed_times


def calculate_reduction(datasets: dict) -> dict[str, dict[str, int]]:
    results: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0))
    )
    for (_, _), somethings_records in datasets.items():
        for record, data_df_by_metric_type in somethings_records:
            for comp_group in ["services", "containers", "middlewares", "nodes"]:
                dfs = data_df_by_metric_type[comp_group]
                for i in range(len(dfs)):
                    results[record.chaos_case_full()][comp_group][str(i)] = dfs[
                        i
                    ].shape[1]
                    results[record.chaos_case_full()]["all"][str(i)] += dfs[i].shape[1]

    avg_reductions: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(lambda: 0)
    )
    for _, result in results.items():
        for comp_group in ["services", "containers", "middlewares", "nodes", "all"]:
            for phase, val in result[comp_group].items():
                avg_reductions[comp_group][phase] += val
    for comp_group in ["services", "containers", "middlewares", "nodes", "all"]:
        for phase in avg_reductions[comp_group]:
            avg_reductions[comp_group][phase] = int(
                avg_reductions[comp_group][phase] / len(results)
            )
    return avg_reductions


def calculate_mean_elapsed_time(
    elapsed_times: dict[tuple[str, str, int], float]
) -> dict:
    df = pd.DataFrame(
        [(_[0], _[1], _[2], elapsed) for _, elapsed in elapsed_times.items()],
        columns=["chaos_type", "chaos_comp", "chaos_case_num", "elapsed_time"],
    )
    return {
        "mean_by_chaos_type-df": neptune.types.File.as_html(
            df.groupby(["chaos_type"]).mean()
        ),
        "mean_by_chaos_comp-df": neptune.types.File.as_html(
            df.groupby(["chaos_comp"]).mean()
        ),
        "mean_by_chaos_type_and_chaos_comp": neptune.types.File.as_html(
            df.groupby(["chaos_type", "chaos_comp"]).mean()
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
    diag_options: dict[str, Any] = DIAG_DEFAULT_OPTIONS,
    use_manually_selected_metrics: bool = False,
    time_range: tuple[int, int] = (0, 0),
    target_chaos_types: set[str] = DEFAULT_CHAOS_TYPES,
    from_orig: tuple[bool, int] = (False, 0),  # from_orig flag, from_orig_num_datapoints
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
    run["reduction"] = calculate_reduction(datasets)

    data_dfs_by_metric_type, elapsed_times = diagnose_and_rank_multi_datasets(
        dataset_id,
        datasets,
        diag_options,
        n_workers=experiment_n_workers,
    )

    run["elapsed_time"] = calculate_mean_elapsed_time(elapsed_times)

    df = create_localization_score_as_dataframe(data_dfs_by_metric_type, pk=pk, k=n)
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
        )
    )
    run["eval/score-df-by-cause-type"] = neptune.types.File.as_html(
        create_localization_score_as_dataframe(
            data_dfs_by_metric_type,
            pk=pk,
            k=n,
            group_by_cause_comp=False,
            group_by_cause_type=True,
        )
    )
    run["eval/score-df-by-cause-comp-and-type"] = neptune.types.File.as_html(
        create_localization_score_as_dataframe(
            data_dfs_by_metric_type,
            pk=pk,
            k=n,
            group_by_cause_comp=True,
            group_by_cause_type=True,
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
    from_orig: tuple[bool, int] = (False, 0),  # from_orig flag, from_orig_num_datapoints
    experiment_id: str = "",
    experiment_n_workers: int = -1,
    progress: bool = False,
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

    params = list(
        itertools.product(
            list_of_diag_options,
            list_of_tsdr_options,
            metric_types_pairs,
            pair_of_use_manually_selected_metrics,
            time_ranges,
        )
    )
    if progress:
        params = tqdm(params)
    for (
        diag_options,
        tsdr_options,
        metric_types,
        use_manually_selected_metrics,
        time_range,
    ) in params:
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
        )
