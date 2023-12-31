from itertools import product
from typing import Any

COST_MODELS = ["l2"]  #, "l1", "normal"]
# SEARCH_METHODS = ["binseg", "pelt", "bottomup"]
SEARCH_METHODS = ["pelt"]
CHANGE_POINTS_PENALTIES = ["bic"]  #, "aic"]
KDE_BANDWIDTHS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
KDE_BANDWIDTH_ADJUSTS = [1.0]

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
        step2_changepoint_segment_selection_method="weighted_max",
        step2_clustering_granularity="service",
    )
    for cost_model, search_method, penalty, kde_bandwidth, kde_bandwidth_adjust
    in product(COST_MODELS, SEARCH_METHODS, CHANGE_POINTS_PENALTIES, KDE_BANDWIDTHS, KDE_BANDWIDTH_ADJUSTS)
]
