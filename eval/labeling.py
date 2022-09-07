from functools import partial
from typing import Any

import pandas as pd

from meltria.loader import DatasetRecord
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule
from tsdr.tsdr import filter_out_no_change_metrics


def detect_anomalies_in_record(
    record: DatasetRecord, labbeling: dict[str, dict[str, Any]], fi_time: int,
) -> pd.DataFrame:
    filtered_df: pd.DataFrame = filter_out_no_change_metrics(record.data_df, parallel=False)

    """ Detect anomalies in a dataset record """
    def detect_anomaly(X: pd.Series, n_sigma: int) -> bool:
        return detect_with_n_sigma_rule(X, test_start_time=fi_time, sigma_threshold=n_sigma).size > 0

    items: list[dict[str, str | int | float]] = []
    for n_sigma in labbeling['n_sigma_rule']['n_sigmas']:
        f = partial(detect_anomaly, n_sigma=n_sigma)
        vc: pd.Series = record.data_df.apply(f).value_counts()
        items.append({
            'target_app': record.target_app(),
            'chaos_type': record.chaos_type(),
            'chaos_comp': record.chaos_comp(),
            'chaos_case_num': record.chaos_case_num(),
            'n_sigma': n_sigma,
            'num_anomalies': f"{vc[True]}/{filtered_df.shape[1]}/{vc[True] + vc[False]}",
            'anomalies_rate': round(vc[True] / filtered_df.shape[1], 2),
        })
    df = pd.DataFrame(
        items,
        columns=['target_app', 'chaos_type', 'chaos_comp', 'chaos_case_num', 'n_sigma', 'num_anomalies', 'anomalies_rate'],
    ).set_index(['target_app', 'chaos_type', 'chaos_comp', 'chaos_case_num', 'n_sigma'])
    return df
