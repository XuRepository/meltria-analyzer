import logging
from enum import IntEnum
from typing import Any, Final

import joblib
import networkx as nx
import pandas as pd
from notebooklib import rank, save

from diagnoser import diag
from meltria.loader import DatasetRecord


class DiagTargetPhaseOption(IntEnum):
    FIRST = -2
    LAST = -1


DEFAULT_DIAG_TARGET_PHASE_OPTION: Final[DiagTargetPhaseOption] = DiagTargetPhaseOption.LAST

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
        root_metric_type=opts.pop("root_metric_type"),
        enable_prior_knowledge=opts.pop("enable_prior_knowledge"),
        **opts,
    )
    if len(ranks) == 0:
        logging.error(f"Failed to diagnose {record.chaos_case_full()} with {len(ranks)} ranks")
        return None
    return G, rank.create_rank_as_dataframe(ranks, dataset_id, record)


def diagnose_and_rank_multi_datasets(
    dataset_id: str,
    datasets: dict,
    n: int,
    diag_options: dict[str, float | bool | int],
    diag_target_phase_option: DiagTargetPhaseOption = DEFAULT_DIAG_TARGET_PHASE_OPTION,
) -> pd.DataFrame:
    assert len(datasets) != 0

    pk = list(datasets.values())[0][0][0].pk
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
    return rank.create_localization_score_as_dataframe([_[1] for _ in results], pk=pk, k=n)


def grid_dataset(
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
        )
        logging.info(f"Processing {dataset_id} with {suffix}...")
        df = diagnose_and_rank_multi_datasets(
            dataset_id, datasets, n, diag_options, diag_target_phase_option=diag_target_phase
        )
        results.append(((suffix, diag_target_phase), df))
    return results
