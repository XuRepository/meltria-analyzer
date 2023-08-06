import logging
from collections import defaultdict

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from diagnoser.metric_node import MetricNodes
from eval.groundtruth import check_cause_metrics
from meltria.loader import DatasetRecord
from meltria.priorknowledge.priorknowledge import PriorKnowledge


def create_rank_as_dataframe(
    ranks: list[tuple[str, float]],
    dataset_id: str,
    record: DatasetRecord,
    n: int = 0,
) -> pd.DataFrame:
    """
    Create a rank as a dataframe from a dictionary of ranks and a dataset.
    """
    items = [
        (
            dataset_id,
            record.target_app(),
            record.chaos_type(),
            record.chaos_comp(),
            record.chaos_case_num(),
            str(metric_name),
            check_cause_metrics(
                record.pk,
                [metric_name],
                record.chaos_type(),
                record.chaos_comp(),
                optional_cause=False,
            )[0],
            check_cause_metrics(
                record.pk,
                [metric_name],
                record.chaos_type(),
                record.chaos_comp(),
                optional_cause=True,
            )[0],
            rank,
            record.data_df[str(metric_name)].values,
        )
        for metric_name, rank in ranks
    ]
    rank_df = pd.DataFrame(
        items,
        columns=[
            "dataset_id",
            "target_app",
            "chaos_type",
            "chaos_comp",
            "chaos_idx",
            "metric_name",
            "cause(mandatory)",
            "cause(optional)",
            "rank",
            "metric_values",
        ],
    ).sort_values("rank", ascending=False)
    if n != 0:
        rank_df = rank_df.head(n=n)
    rank_df.reset_index(inplace=True)
    rank_df.index += 1
    return rank_df


def create_rank_as_dataframe_for_multiple_cases(
    ranks_of_cases: list[list[tuple[str, float]]],
    dataset_id: str,
    records: list[DatasetRecord],
    n: int = 0,
) -> pd.DataFrame:
    list_of_rank_df = [
        create_rank_as_dataframe(ranks, dataset_id, records[i], n=n)
        for i, ranks in enumerate(ranks_of_cases)
    ]
    return create_rank_as_dataframe_for_multiple_cases_from_frames(list_of_rank_df, n=n)


def create_rank_as_dataframe_for_multiple_cases_from_frames(
    list_of_rank_df: list[pd.DataFrame], n: int = 0
) -> pd.DataFrame:
    ranks_df = pd.concat(list_of_rank_df, axis=0)
    if n == 0:
        return ranks_df
    return (
        _sort_and_group_by(ranks_df)
        .head(n=n)
        .set_index(
            ["dataset_id", "target_app", "chaos_type", "chaos_comp", "chaos_idx"]
        )
    )


def _sort_and_group_by(ranks_df: pd.DataFrame) -> DataFrameGroupBy:
    return (
        ranks_df.loc[
            :,
            [
                "dataset_id",
                "target_app",
                "chaos_type",
                "chaos_comp",
                "chaos_idx",
                "metric_name",
                "rank",
            ],
        ]
        .dropna(subset=["rank"])
        .sort_values(
            [
                "dataset_id",
                "target_app",
                "chaos_type",
                "chaos_comp",
                "chaos_idx",
                "rank",
            ],
            ascending=False,
        )
        .groupby(
            ["dataset_id", "target_app", "chaos_type", "chaos_comp", "chaos_idx"],
            as_index=True,
        )
    )


def create_localization_score_as_dataframe(
    list_of_rank_df: list[pd.DataFrame],
    pk: PriorKnowledge,
    k: int = 10,
    group_by_cause_type: bool = False,
    group_by_cause_comp: bool = False,
) -> pd.DataFrame:
    ranks_df = create_rank_as_dataframe_for_multiple_cases_from_frames(list_of_rank_df)
    scores_df: pd.DataFrame
    match (group_by_cause_type, group_by_cause_comp):
        case (False, False):
            scores_df = pd.concat(
                [
                    evaluate_ac_of_rc(
                        ranks_df.groupby(
                            [
                                "dataset_id",
                                "target_app",
                                "chaos_type",
                                "chaos_comp",
                                "chaos_idx",
                            ]
                        ),
                        pk,
                        granuallity=gran,
                        k=k,
                    )
                    for gran in ["metric", "container", "service"]
                ],
                axis=1,
            )
        case (True, False):
            dfs: list[pd.DataFrame] = []
            chaos_types: list[tuple] = []
            for chaos_type, group in ranks_df.groupby(
                ["dataset_id", "target_app", "chaos_type"]
            ):
                dfs.append(
                    pd.concat(
                        [
                            evaluate_ac_of_rc(
                                group.groupby(
                                    [
                                        "dataset_id",
                                        "target_app",
                                        "chaos_type",
                                        "chaos_comp",
                                        "chaos_idx",
                                    ]
                                ),
                                pk,
                                granuallity=gran,
                                k=k,
                            )
                            for gran in ["metric", "container", "service"]
                        ],
                        axis=1,
                    )
                )
                chaos_types.append(chaos_type)
            scores_df = pd.concat(dfs, axis=0, keys=chaos_types)
        case (False, True):
            dfs: list[pd.DataFrame] = []
            chaos_comps: list[tuple] = []
            for chaos_comp, group in ranks_df.groupby(
                ["dataset_id", "target_app", "chaos_comp"]
            ):
                dfs.append(
                    pd.concat(
                        [
                            evaluate_ac_of_rc(
                                group.groupby(
                                    [
                                        "dataset_id",
                                        "target_app",
                                        "chaos_type",
                                        "chaos_comp",
                                        "chaos_idx",
                                    ]
                                ),
                                pk,
                                granuallity=gran,
                                k=k,
                            )
                            for gran in ["metric", "container", "service"]
                        ],
                        axis=1,
                    )
                )
                chaos_comps.append(chaos_comp)
            scores_df = pd.concat(dfs, axis=0, keys=chaos_comps)
        case (True, True):
            dfs: list[pd.DataFrame] = []
            agg_keys: list[tuple] = []
            for chaos_comp, group in ranks_df.groupby(
                ["dataset_id", "target_app", "chaos_comp", "chaos_type"]
            ):
                dfs.append(
                    pd.concat(
                        [
                            evaluate_ac_of_rc(
                                group.groupby(
                                    [
                                        "dataset_id",
                                        "target_app",
                                        "chaos_type",
                                        "chaos_comp",
                                        "chaos_idx",
                                    ]
                                ),
                                pk,
                                granuallity=gran,
                                k=k,
                            )
                            for gran in ["metric", "container", "service"]
                        ],
                        axis=1,
                    )
                )
                agg_keys.append(chaos_comp)
            scores_df = pd.concat(dfs, axis=0, keys=agg_keys)
        case _:
            raise ValueError(
                f"Invalid combination of group_by_cause_type and group_by_cause_comp: {group_by_cause_type}, {group_by_cause_comp}"
            )
    return scores_df


def get_ranks_by_case(
    sorted_results_df: DataFrameGroupBy,
    pk: PriorKnowledge,
    granularity: str = "metric",
    optional_cause: bool = True,
) -> dict[tuple[str, str, int], list[int]]:
    ranks_by_case: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for (
        dataset_id,
        target_app,
        chaos_type,
        chaos_comp,
        chaos_case_num,
    ), row in sorted_results_df:
        if chaos_comp in pk.get_skip_containers():
            continue
        metrics = [str(m) for m in row["metric_name"].values.tolist()]
        ranks: list[int]
        match granularity:
            case "metric":
                ok, cause_metrics = check_cause_metrics(
                    pk,
                    metrics,
                    chaos_type=chaos_type,
                    chaos_comp=chaos_comp,
                    optional_cause=optional_cause,
                )
                if not ok or len(cause_metrics) == 0:
                    logging.info(
                        f"no cause metrics: {dataset_id}, {target_app}, {chaos_type}/{chaos_comp}/{chaos_case_num}"
                    )
                    ranks_by_case[(chaos_type, chaos_comp, chaos_case_num)] = []
                    continue
                metrics = [
                    m for m in metrics if not m.startswith("s-")
                ]  # Exclude service metrics
                ranked_metrics = MetricNodes.from_metric_names(metrics)
                ranks = sorted(
                    [list(ranked_metrics).index(cm) + 1 for cm in cause_metrics]
                )
            case "container":
                metrics = [
                    m for m in metrics if not m.startswith("s-")
                ]  # Exclude service metrics
                ranked_ctnrs = list(
                    dict.fromkeys(
                        [pk.get_container_by_metric(metric) for metric in metrics]
                    )
                )
                ranks = sorted(
                    [i + 1 for i, ctnr in enumerate(ranked_ctnrs) if ctnr == chaos_comp]
                )
            case "service":
                chaos_service: str = pk.get_service_by_container(chaos_comp)
                ranked_service = list(
                    dict.fromkeys(
                        [pk.get_service_by_metric(metric) for metric in metrics]
                    )
                )
                ranked_service = [
                    s
                    for s in ranked_service
                    if s is not None and not s.startswith("gke-")
                ]
                ranks = sorted(
                    [
                        i + 1
                        for i, service in enumerate(ranked_service)
                        if service == chaos_service
                    ]
                )
            case _:
                assert False, f"Unknown detect_unit: {granularity}"
        ranks_by_case[(chaos_type, chaos_comp, chaos_case_num)] = ranks
    return ranks_by_case


def ac_k_for_any_cause_metrics(
    k: int, cause_ranks_by_case: dict[tuple[str, str, int], list[int]]
) -> float:
    sum_ac: float = 0.0
    num_anomalies: int = len(cause_ranks_by_case.keys())
    for _, cause_ranks in cause_ranks_by_case.items():
        # "/ k" should be "/ min(k, num_root_cause_metrics)"
        num_correct = any([rank <= k for rank in cause_ranks])
        sum_ac += num_correct
    return sum_ac / num_anomalies


def ac_k_for_all_cause_metrics(
    k: int, cause_ranks_by_case: dict[tuple[str, str, int], list[int]]
) -> float:
    sum_ac: float = 0.0
    num_anomalies: int = len(cause_ranks_by_case.keys())
    for _, cause_ranks in cause_ranks_by_case.items():
        # "/ k" should be "/ min(k, num_root_cause_metrics)"
        num_correct = sum([1 for rank in cause_ranks if rank <= k])
        sum_ac += num_correct / k
    return sum_ac / num_anomalies


def evaluate_ac_of_rc(
    sorted_results_df: DataFrameGroupBy,
    pk: PriorKnowledge,
    k: int = 10,
    granuallity: str = "metric",
) -> pd.DataFrame:
    top_k_set = range(1, k + 1)

    ranks_by_case = get_ranks_by_case(
        sorted_results_df, pk, granularity=granuallity, optional_cause=True
    )
    ac_k = {k: ac_k_for_any_cause_metrics(k, ranks_by_case) for k in top_k_set}
    avg_k = {k: sum([ac_k[j] for j in range(1, k + 1)]) / k for k in top_k_set}

    ranks_by_case_mand = get_ranks_by_case(
        sorted_results_df, pk, granularity=granuallity, optional_cause=False
    )
    ac_k_mand = {
        k: ac_k_for_any_cause_metrics(k, ranks_by_case_mand) for k in top_k_set
    }
    avg_k_mand = {
        k: sum([ac_k_mand[j] for j in range(1, k + 1)]) / k for k in top_k_set
    }

    return pd.concat(
        [
            pd.DataFrame(
                {k: len(ranks_by_case.keys()) for k in top_k_set},
                index=[f"#cases ({granuallity})"],
            ).T,
            pd.DataFrame(ac_k, index=[f"AC@K ({granuallity})"]).T,
            pd.DataFrame(avg_k, index=[f"AVG@K ({granuallity})"]).T,
            pd.DataFrame(ac_k_mand, index=[f"AC@K ({granuallity}) (mandatory)"]).T,
            pd.DataFrame(avg_k_mand, index=[f"AVG@K ({granuallity}) (mandatory)"]).T,
        ],
        axis=1,
    )
