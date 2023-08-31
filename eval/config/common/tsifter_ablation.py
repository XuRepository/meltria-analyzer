from typing import Any

TSDR_OPTIONS: list[dict[str, Any]] = [
    dict(  # TSifter Ablation
        enable_unireducer=False,
        enable_multireducer=True,
        step1_method_name="changepoint",
        step1_changepoint_search_method="binseg",
        step1_changepoint_cost_model="l2",
        step1_changepoint_penalty="bic",
    )
]
