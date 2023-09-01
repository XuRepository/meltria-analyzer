from eval.config.common.tsifter_best_params import TSIFTER_BEST_PARAM
from eval.config.sweep_localization.common.all_localizations import \
    LIST_OF_DIAG_OPTIONS

CONFIG = dict(
    dataset_id="9n6mf",
    n=10,
    experiment_n_workers=-1,
    target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
    list_of_tsdr_options=[TSIFTER_BEST_PARAM],
    list_of_diag_options=LIST_OF_DIAG_OPTIONS,
    pair_of_use_manually_selected_metrics=[True],
    metric_types_pairs=[{
        "services": True,
        "containers": True,
        "middlewares": False,
        "nodes": False,
    }],
    sampling_scale_factors=[1, 2, 3, 4],
    progress=True,
    timeout_sec=60 * 60 * 1,
    # from_orig=(True, 180, 20, 3),
)
