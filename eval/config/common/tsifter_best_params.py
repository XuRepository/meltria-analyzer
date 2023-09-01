from typing import Any

TSIFTER_BEST_PARAM: dict[str, Any] = dict(
    enable_unireducer=False,
    enable_multireducer=True,
    step2_clustering_method_name="changepoint-kde",
    step2_changepoint_multi_change_points=True,
    step2_changepoint_n_bkps=1,
    step2_changepoint_search_method="binseg",
    step2_changepoint_cost_model="l2",
    step2_changepoint_penalty="bic",
    step2_changepoint_kde_bandwidth="scott",
    step2_changepoint_representative_method=False,
    step2_clustering_n_workers=1,
)
