from eval.config.common.tsdr_options import TSDR_OPTIONS
from eval.config.sweep_localization.common.localizations import \
    get_list_of_diag_options

CONFIG = dict(
    dataset_id="m9dgg",
    n=10,
    experiment_n_workers=-1,
    target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
    list_of_tsdr_options=TSDR_OPTIONS,
    list_of_diag_options=get_list_of_diag_options(mode="fast-only"),
    pair_of_use_manually_selected_metrics=[False],
    metric_types_pairs=[{
        "services": True,
        "containers": True,
        "middlewares": False,
        "nodes": False,
    }],
    # from_orig=(True, 180, 20, 3),
    progress=True,
    timeout_sec=60 * 60 * 1,
)
