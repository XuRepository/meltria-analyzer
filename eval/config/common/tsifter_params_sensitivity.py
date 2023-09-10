from itertools import product
from typing import Any

COST_MODELS = ["l2", "normal"]
SEARCH_METHODS = ["binseg", "pelt", "bottomup"]
CHANGE_POINTS_PENALTIES = ["bic", "aic"]
KDE_BANDWIDTHS = ["scott", "silverman"]
KDE_BANDWIDTH_ADJUSTS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

TSDR_OPTIONS: list[dict[str, Any]] = [
    dict(
        enable_unireducer=True,
        enable_multireducer=True,
        step1_method_name="changepoint",
        step1_changepoint_search_method=search_method,
        step1_changepoint_cost_model=cost_model,
        step1_changepoint_penalty=penalty,
        step2_clustering_method_name="changepoint-kde",
        step2_changepoint_kde_bandwidth=kde_bandwidth,
        step2_changepoint_kde_bandwidth_adjust=kde_bandwidth_adjust,
        step2_clustering_granularity="service",
    )
    for cost_model, search_method, penalty, kde_bandwidth, kde_bandwidth_adjust
    in product(COST_MODELS, SEARCH_METHODS, CHANGE_POINTS_PENALTIES, KDE_BANDWIDTHS, KDE_BANDWIDTH_ADJUSTS)
]
