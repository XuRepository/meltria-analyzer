from collections import defaultdict
from multiprocessing import cpu_count
from typing import Any

import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import zscore
from threadpoolctl import threadpool_limits


def localize(dataset: pd.DataFrame, **kwargs: Any) -> list[tuple[str, float]]:
    import utils as rcdutils

    import rcd

    # _remove_last_seen = lambda df: df.loc[:, ~df.columns.str.endswith('last_seen')]
    # dataset = _remove_last_seen(dataset).apply(zscore).dropna(how="any", axis=1)
    dataset = dataset.apply(zscore).dropna(how="any", axis=1)
    normal_df = dataset[dataset.index < kwargs["rcd_boundary_index"]]
    anomalous_df = dataset[dataset.index >= kwargs["rcd_boundary_index"]]

    df = rcdutils.add_fnode(normal_df, anomalous_df)
    n_df = df[df[rcdutils.F_NODE] == "0"].drop(columns=[rcdutils.F_NODE])
    a_df = df[df[rcdutils.F_NODE] == "1"].drop(columns=[rcdutils.F_NODE])

    n_iters: int = kwargs["rcd_n_iters"]  # for random seed ensamble
    bins: int = kwargs["rcd_bins"]
    localized: bool = kwargs["rcd_localized"]
    gamma: int = kwargs["rcd_gamma"]
    k: int = kwargs["rcd_topk"]
    n_workers: int = kwargs["rcd_n_workers"]
    n_workers_for_seed_ensamble: int = kwargs["rcd_n_workers_seed_ensamble"]

    # predict cause metrics with random seed ensamble because pf the randomness of phi-PC in RCD
    def run_rcd() -> dict[str, Any]:
        with threadpool_limits(limits=1):
            return rcd.rca_with_rcd(n_df, a_df, bins=bins, gamma=gamma, localized=localized, n_workers=n_workers)

    if n_iters < cpu_count():
        if n_workers_for_seed_ensamble == -1:
            n_workers_for_seed_ensamble = n_iters

    results: list[dict[str, Any]]
    if n_workers_for_seed_ensamble == 1:
        results = [run_rcd() for _ in range(n_iters)]
    else:
        results = Parallel(n_jobs=n_workers_for_seed_ensamble)(delayed(run_rcd)() for _ in range(n_iters))
        assert results is not None, "The results of rcd.rca_with_rcd are not empty"

    scores: dict[str, int] = defaultdict(int)
    for result in results:
        for m in result["root_cause"][:k]:
            scores[m] += 1
    return sorted([(metric, n / n_iters) for (metric, n) in scores.items()], key=lambda x: x[1], reverse=True)
