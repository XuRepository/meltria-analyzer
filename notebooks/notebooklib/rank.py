import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from eval.groundtruth import check_cause_metrics
from eval.localizaiton_score import evaluate_ac_of_rc
from meltria.loader import DatasetRecord
from meltria.priorknowledge.priorknowledge import PriorKnowledge


def create_rank_as_dataframe(
    ranks: list[tuple[str, float]], dataset_id: str, record: DatasetRecord, n: int = 10
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
                record.pk, [metric_name], record.chaos_type(), record.chaos_comp(), optional_cause=False
            )[0],
            check_cause_metrics(
                record.pk, [metric_name], record.chaos_type(), record.chaos_comp(), optional_cause=True
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
    n: int = 10,
) -> pd.DataFrame:
    list_of_rank_df = [
        create_rank_as_dataframe(ranks, dataset_id, records[i], n=n) for i, ranks in enumerate(ranks_of_cases)
    ]
    return create_rank_as_dataframe_for_multiple_cases_from_frames(list_of_rank_df, n=n)


def create_rank_as_dataframe_for_multiple_cases_from_frames(
    list_of_rank_df: list[pd.DataFrame], n: int = 10
) -> pd.DataFrame:
    ranks_df = pd.concat(list_of_rank_df, axis=0)
    return (
        _sort_and_group_by(ranks_df)
        .head(n=n)
        .set_index(["dataset_id", "target_app", "chaos_type", "chaos_comp", "chaos_idx"])
    )


def _sort_and_group_by(ranks_df: pd.DataFrame) -> DataFrameGroupBy:
    return (
        ranks_df.loc[:, ["dataset_id", "target_app", "chaos_type", "chaos_comp", "chaos_idx", "metric_name", "rank"]]
        .dropna(subset=["rank"])
        .sort_values(["dataset_id", "target_app", "chaos_type", "chaos_comp", "chaos_idx", "rank"], ascending=False)
        .groupby(["dataset_id", "target_app", "chaos_type", "chaos_comp", "chaos_idx"], as_index=True)
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
                    evaluate_ac_of_rc(ranks_df, pk, granuallity=gran, k=k)
                    for gran in ["metric", "container", "service"]
                ],
                axis=1,
            )
        case (True, False):
            dfs: list[pd.DataFrame] = []
            chaos_types: list[tuple] = []
            for chaos_type, group in ranks_df.groupby(["dataset_id", "target_app", "chaos_type"]):
                dfs.append(
                    pd.concat(
                        [
                            evaluate_ac_of_rc(
                                group.groupby(["dataset_id", "target_app", "chaos_type", "chaos_comp", "chaos_idx"]),
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
            for chaos_comp, group in ranks_df.groupby(["dataset_id", "target_app", "chaos_comp"]):
                dfs.append(
                    pd.concat(
                        [
                            evaluate_ac_of_rc(
                                group.groupby(["dataset_id", "target_app", "chaos_type", "chaos_comp", "chaos_idx"]),
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
            for chaos_comp, group in ranks_df.groupby(["dataset_id", "target_app", "chaos_comp", "chaos_type"]):
                dfs.append(
                    pd.concat(
                        [
                            evaluate_ac_of_rc(
                                group.groupby(["dataset_id", "target_app", "chaos_type", "chaos_comp", "chaos_idx"]),
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
    return scores_df
