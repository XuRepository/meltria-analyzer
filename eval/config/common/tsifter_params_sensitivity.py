from itertools import product
from typing import Any

CHANGE_POINTS_PENALTIES = ["bic", "aic"]
KDE_BANDWIDTHS = ["silverman", "scott", 1.0]
COST_MODELS = ["l2", "normal"]

TSDR_OPTIONS: list[dict[str, Any]] = [
    dict(
        enable_unireducer=False,
        enable_multireducer=True,
        step2_clustering_method_name="changepoint-kde",
        step2_changepoint_multi_change_points=True,
        step2_changepoint_n_bkps=1,
        step2_changepoint_penalty=penalty,
        step2_changepoint_kde_bandwidth=kde_bandwidth,
        step2_changepoint_representative_method=False,
        step2_changepoint_cost_model=cost_model,
        step2_clustering_n_workers=1,
    )
    for penalty, kde_bandwidth, cost_model in product(CHANGE_POINTS_PENALTIES, KDE_BANDWIDTHS, COST_MODELS)
]
