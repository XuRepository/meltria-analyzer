import pandas as pd

from meltria import loader


def test_count_metrics():
    df = pd.DataFrame(
        {
            "s-front-end_latency": [],
            "s-front-end_throughput": [],
            "c-front-end_cpu_usage": [],
            "c-front-end_memory_usage": [],
            "s-user_latency": [],
            "s-user_throughput": [],
            "c-user_cpu_usage": [],
            "c-user_memory_usage": [],
            "c-user-db_cpu_usage": [],
            "c-user-db_memory_usage": [],
        }
    )
    count_df = loader.count_metrics(df)
    assert count_df.loc[("services", "front-end"), "count"] == 2
    assert count_df.loc[("containers", "front-end"), "count"] == 2
    assert count_df.loc[("services", "user"), "count"] == 2
    assert count_df.loc[("containers", "user"), "count"] == 2
    assert count_df.loc[("containers", "user-db"), "count"] == 2
