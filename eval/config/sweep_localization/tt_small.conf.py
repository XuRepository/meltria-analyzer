from eval.config.common.tsdr_options import TSDR_OPTIONS
from eval.config.sweep_localization.common.fast_localizations import \
    LIST_OF_DIAG_OPTIONS

CONFIG = dict(
    dataset_id="9n6mf",
    n=10,
    experiment_n_workers=-1,
    target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
    list_of_tsdr_options=TSDR_OPTIONS,
    list_of_diag_options=LIST_OF_DIAG_OPTIONS,
    pair_of_use_manually_selected_metrics=[True],
    metric_types_pairs=[{
        "services": True,
        "containers": True,
        "middlewares": False,
        "nodes": False,
    }],
    progress=True,
    timeout_sec=60 * 60 * 1,
)
