import sys
from collections import defaultdict
from typing import Any

import pandas as pd
from scipy.stats import zscore
from threadpoolctl import threadpool_limits

sys.path.append("../rcd")
import utils as rcdutils

import rcd


def localize(dataset: pd.DataFrame, **kwargs: Any) -> list[tuple[str, float]]:
    # _remove_last_seen = lambda df: df.loc[:, ~df.columns.str.endswith('last_seen')]
    # dataset = _remove_last_seen(dataset).apply(zscore).dropna(how="any", axis=1)
    dataset = dataset.apply(zscore).dropna(how="any", axis=1)
    normal_df = dataset[dataset.index < kwargs["rcd_boundary_index"]]
    anomalous_df = dataset[dataset.index >= kwargs["rcd_boundary_index"]]

    df = rcdutils.add_fnode(normal_df, anomalous_df)
    n_df = df[df[rcdutils.F_NODE] == "0"].drop(columns=[rcdutils.F_NODE])
    a_df = df[df[rcdutils.F_NODE] == "1"].drop(columns=[rcdutils.F_NODE])

    n_iters: int = kwargs["rcd_n_iters"]
    bins: int = kwargs["rcd_bins"]
    localized: bool = kwargs["rcd_localized"]
    k: int = kwargs["rcd_topk"]
    results: dict[str, int] = defaultdict(int)
    for i in range(n_iters):
        with threadpool_limits(limits=1):
            result = rcd.rca_with_rcd(n_df, a_df, bins=bins, gamma=kwargs["rcd_gamma"], localized=localized)
        for m in result["root_cause"][:k]:
            results[m] += 1
    return sorted([(metric, n / n_iters) for (metric, n) in results.items()], key=lambda x: x[1], reverse=True)
