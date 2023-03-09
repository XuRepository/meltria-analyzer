import pandas as pd

from meltria.loader import DatasetRecord


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
            record.data_df[str(metric_name)].values,
            rank,
        )
        for metric_name, rank in ranks
    ]
    rank_df = (
        pd.DataFrame(
            items,
            columns=[
                "dataset_id",
                "target_app",
                "chaos_type",
                "chaos_comp",
                "chaos_case_num",
                "metric_name",
                "metric_values",
                "rank",
            ],
        )
        .sort_values("rank", ascending=False)
        .head(n=n)
        .reset_index()
    )
    rank_df.index += 1
    return rank_df
