import datetime
import itertools
import logging
import os
import warnings
from collections import defaultdict
from enum import IntEnum
from typing import Any, Final

import joblib
import neptune
import networkx as nx
import pandas as pd

from diagnoser import diag
from eval.localizaiton_score import create_localization_score_as_dataframe, create_rank_as_dataframe
from eval.tsdr import METRIC_TYPES_PAIRS, check_cache_suffix, load_tsdr_by_chaos
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
        **opts,
    )
    if len(ranks) == 0:
        logging.error(f"Failed to diagnose {record.chaos_case_full()} with {len(ranks)} ranks")
        return None
    return G, create_rank_as_dataframe(ranks, dataset_id, record)


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


def load_tsdr_and_localize(
    experiment_id: str,
    dataset_id: str,
    n: int,
    metric_types: dict[str, bool],
    tsdr_options: dict[str, Any],
    diag_options: dict[str, Any] = DIAG_DEFAULT_OPTIONS,
    use_manually_selected_metrics: bool = False,
) -> None:
    datasets = load_tsdr_by_chaos(
        dataset_id,
        metric_types=metric_types,
        revert_normalized_time_series=True,
        tsdr_options=tsdr_options,
        use_manually_selected_metrics=use_manually_selected_metrics,
    )

    first_record = list(datasets.values())[0][0][0]
    pk = first_record.pk

    run = neptune.init_run(
        project=os.environ["TSDR_LOCALIZATION_NEPTUNE_PROJECT"],
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )
    run["experiment_id"] = experiment_id
    run["dataset"] = {
        "dataset_id": dataset_id,
        "target_app": first_record.target_app(),
        "metric_types": metric_types,
        "use_manually_selected_metrics": use_manually_selected_metrics,
    }
    run["parameters/tsdr"] = tsdr_options
    run["parameters"] = diag_options
    run["reduction"] = calculate_reduction(datasets)

    data_dfs_by_metric_type = diagnose_and_rank_multi_datasets(
        dataset_id,
        datasets,
        diag_options,
    )

    df = create_localization_score_as_dataframe(data_dfs_by_metric_type, pk=pk, k=n)
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


def sweep_localization_and_save_as_cache(
    dataset_id: str,
    n: int,
    list_of_tsdr_options: list[dict[str, Any]],
    list_of_diag_options: list[dict[str, Any]],
    pair_of_use_manually_selected_metrics: list[bool],
    experiment_id: str = "",
) -> None:
    if experiment_id == "":
        experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    for tsdr_options, metric_types, use_manually_selected_metrics in itertools.product(
        list_of_tsdr_options, METRIC_TYPES_PAIRS, pair_of_use_manually_selected_metrics
    ):
        if not check_cache_suffix(dataset_id, metric_types, tsdr_options, use_manually_selected_metrics):
            raise ValueError(
                f"None of cache is available for {metric_types}, use_manually_selected_metric={use_manually_selected_metrics}, {tsdr_options}"
            )

    for diag_options, tsdr_options, metric_types, use_manually_selected_metrics in itertools.product(
        list_of_diag_options,
        list_of_tsdr_options,
        METRIC_TYPES_PAIRS,
        pair_of_use_manually_selected_metrics,
    ):
        logging.info(
            f"Processing {metric_types}, manually_selected?={use_manually_selected_metrics}, {tsdr_options}, {diag_options}"
        )
        load_tsdr_and_localize(
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            n=n,
            metric_types=metric_types,
            tsdr_options=tsdr_options,
            diag_options=diag_options,
            use_manually_selected_metrics=use_manually_selected_metrics,
        )
