import datetime
import logging
import os
import warnings
from collections import defaultdict
from enum import IntEnum
from typing import Any, Final

import joblib
import neptune
import networkx as nx
import numpy as np
import pandas as pd
from notebooklib import rank, save

from diagnoser import diag
from meltria.loader import DatasetRecord

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


class DiagTargetPhaseOption(IntEnum):
    FIRST = -2
    LAST = -1
    RAW = 0


DEFAULT_DIAG_TARGET_PHASE_OPTION: Final[DiagTargetPhaseOption] = DiagTargetPhaseOption.LAST
DEFAULT_RESULT_GROUPBY: Final[dict[str, bool]] = dict(group_by_cause_type=False, group_by_cause_comp=False)

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


def diagnose_and_rank(
    dataset_id: str,
    reduced_df: pd.DataFrame,
    record: DatasetRecord,
    diag_options: dict[str, str | float | bool] = DIAG_DEFAULT_OPTIONS,
) -> tuple[nx.Graph, pd.DataFrame] | None:
    opts = dict(DIAG_DEFAULT_OPTIONS, **diag_options)
    G, ranks = diag.build_and_walk_causal_graph(
        reduced_df,
        record.pk,
        # root_metric_type=opts.pop("root_metric_type"),
        # enable_prior_knowledge=opts.pop("enable_prior_knowledge"),
        # use_call_graph=opts.pop("use_call_graph"),
        # use_complete_graph=opts.pop("use_complete_graph"),
        **opts,
    )
    if len(ranks) == 0:
        logging.error(f"Failed to diagnose {record.chaos_case_full()} with {len(ranks)} ranks")
        return None
    return G, rank.create_rank_as_dataframe(ranks, dataset_id, record)


def diagnose_and_rank_multi_datasets(
    dataset_id: str,
    datasets: dict,
    diag_options: dict[str, float | bool | int],
    diag_target_phase_option: DiagTargetPhaseOption = DEFAULT_DIAG_TARGET_PHASE_OPTION,
) -> list:
    assert len(datasets) != 0

    records = []
    for (_, _), somethings_records in datasets.items():
        for record, data_df_by_metric_type in somethings_records:
            reduced_df = pd.concat(
                [
                    data_df_by_metric_type["services"][diag_target_phase_option.value],
                    data_df_by_metric_type["containers"][diag_target_phase_option.value],
                    data_df_by_metric_type["middlewares"][diag_target_phase_option.value],
                    data_df_by_metric_type["nodes"][diag_target_phase_option],
                ],
                axis=1,
            )
            if not any(label in reduced_df.columns for label in record.pk.get_root_metrics()):
                logging.warn(f"No root metrics found in the dataset {record.chaos_case_full()}: {reduced_df.shape[1]}")
                continue
            records.append((record, reduced_df))

    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(diagnose_and_rank)(dataset_id, reduced_df, record, diag_options)
        for record, reduced_df in records
    )
    results = [result for result in results if result is not None]
    assert results is not None
    assert len(results) != 0
    return [_[1] for _ in results]


def calculate_reduction(datasets: dict) -> dict[str, dict[str, int]]:
    results: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    for (_, _), somethings_records in datasets.items():
        for record, data_df_by_metric_type in somethings_records:
            for comp_group in ["services", "containers", "middlewares", "nodes"]:
                dfs = data_df_by_metric_type[comp_group]
                for i in range(len(dfs)):
                    results[record.chaos_case_full()][comp_group][str(i)] = dfs[i].shape[1]
                    results[record.chaos_case_full()]["all"][str(i)] += dfs[i].shape[1]

    avg_reductions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for _, result in results.items():
        for comp_group in ["services", "containers", "middlewares", "nodes", "all"]:
            for phase, val in result[comp_group].items():
                avg_reductions[comp_group][phase] += val
    for comp_group in ["services", "containers", "middlewares", "nodes", "all"]:
        for phase in avg_reductions[comp_group]:
            avg_reductions[comp_group][phase] = int(avg_reductions[comp_group][phase] / len(results))
    return avg_reductions


def wrap_diagnose_with_neptune(
    expriment_id: str,
    dataset_id: str,
    datasets: dict,
    n: int,
    diag_options: dict[str, float | bool | int],
    suffix: str,
    diag_target_phase_option: DiagTargetPhaseOption = DEFAULT_DIAG_TARGET_PHASE_OPTION,
) -> list:
    first_record = list(datasets.values())[0][0][0]
    pk = first_record.pk

    run = neptune.init_run(
        project=os.environ["TSDR_LOCALIZATION_NEPTUNE_PROJECT"],
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )
    run["experiment_id"] = expriment_id
    run["dataset"] = {
        "dataset_id": dataset_id,
        "target_app": first_record.target_app(),
        "suffix": suffix,
    }
    run["parameters"] = diag_options
    run["reduction"] = calculate_reduction(datasets)

    data_dfs_by_metric_type = diagnose_and_rank_multi_datasets(
        dataset_id,
        datasets,
        diag_options,
        diag_target_phase_option=diag_target_phase_option,
    )

    df = rank.create_localization_score_as_dataframe(data_dfs_by_metric_type, pk=pk, k=n)
    run["scores/metric/num_cases"] = df.at[1, "#cases (metric)"]
    run["scores/container/num_cases"] = df.at[1, "#cases (container)"]
    run["scores/service/num_cases"] = df.at[1, "#cases (service)"]
    for ind, row in df.iterrows():
        for name in ["AC", "AVG"]:
            run[f"scores/metric/{name}_{ind}"] = row[f"{name}@K (metric)"]
            run[f"scores/container/{name}_{ind}"] = row[f"{name}@K (container)"]
            run[f"scores/service/{name}_{ind}"] = row[f"{name}@K (service)"]
    run["eval/score-df"] = neptune.types.File.as_html(df)
    run["eval/score-df-by-cause-comp"] = neptune.types.File.as_html(
        rank.create_localization_score_as_dataframe(
            data_dfs_by_metric_type,
            pk=pk,
            k=n,
            group_by_cause_comp=True,
            group_by_cause_type=False,
        )
    )
    run["eval/score-df-by-cause-type"] = neptune.types.File.as_html(
        rank.create_localization_score_as_dataframe(
            data_dfs_by_metric_type,
            pk=pk,
            k=n,
            group_by_cause_comp=False,
            group_by_cause_type=True,
        )
    )
    run["eval/score-df-by-cause-comp-and-type"] = neptune.types.File.as_html(
        rank.create_localization_score_as_dataframe(
            data_dfs_by_metric_type,
            pk=pk,
            k=n,
            group_by_cause_comp=True,
            group_by_cause_type=True,
        )
    )
    run.stop()

    return data_dfs_by_metric_type


def grid_dataset(
    expriment_id: str,
    dataset_cache_suffixs: list[str | tuple[str, DiagTargetPhaseOption]],
    dataset_id: str,
    n: int,
    diag_options: dict[str, Any] = DIAG_DEFAULT_OPTIONS,
) -> list[tuple[tuple[str, DiagTargetPhaseOption], pd.DataFrame]]:
    for i, item in enumerate(dataset_cache_suffixs):
        if type(item) == str:
            dataset_cache_suffixs[i] = (item, DEFAULT_DIAG_TARGET_PHASE_OPTION)

    if not all([save.check_cache_suffix(dataset_id, suffix) for suffix, _ in dataset_cache_suffixs]):
        raise ValueError(f"None of {dataset_cache_suffixs} is available for {dataset_id}")

    results: list[tuple[tuple[str, DiagTargetPhaseOption], pd.DataFrame]] = []
    for suffix, diag_target_phase in dataset_cache_suffixs:
        logging.info(f"Loading {dataset_id} with {suffix}...")
        datasets = save.load_tsdr_by_chaos(
            dataset_id,
            revert_normalized_time_series=True,
            suffix=suffix,
            manually_selected=use_manually_selected_metrics,
        )
        logging.info(f"Processing {dataset_id} with {suffix}...")
        df = wrap_diagnose_with_neptune(
            expriment_id,
            dataset_id,
            datasets,
            n=n,
            diag_options=diag_options,
            suffix=suffix,
            diag_target_phase_option=diag_target_phase,
        )
        results.append(((suffix, diag_target_phase), df))

    return results


def grid_dataset_with_multi_diag_options(
    dataset_cache_suffixs: list[str | tuple[str, DiagTargetPhaseOption]],
    dataset_id: str,
    n: int,
    list_of_diag_options: list[dict[str, Any]],
    experiment_id: str = "",
) -> list[tuple[tuple[str, DiagTargetPhaseOption, dict], pd.DataFrame]]:
    if experiment_id == "":
        experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    results = []
    for diag_options in list_of_diag_options:
        res = grid_dataset(
            experiment_id, dataset_cache_suffixs, dataset_id, n, diag_options,
        )
        for (suffix, diag_target_phase), df in res:
            results.append(((suffix, diag_target_phase, diag_options), df))
    return results
