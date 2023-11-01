from typing import Any

TSIFTER_BEST_PARAM: dict[str, Any] = dict(
    enable_unireducer=True,
    enable_multireducer=True,
    step1_method_name="changepoint",
    step1_changepoint_search_method="pelt",
    step1_changepoint_cost_model="l2",
    step1_changepoint_penalty="bic",
    step2_clustering_method_name="changepoint-kde",
    step2_changepoint_kde_bandwidth=2.5,
    step2_changepoint_kde_bandwidth_adjust=1.0,
    step2_changepoint_segment_selection_method="weighted_max",
    step2_clustering_granularity="service",
)
