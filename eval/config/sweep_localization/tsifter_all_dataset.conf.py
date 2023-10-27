from eval.config.common.tsdr_options import get_tsifter_options
from eval.config.sweep_localization.common.localizations import \
    get_list_of_diag_options

DEFAULT_TIMEOUT_SEC = 60 * 60 * 1

CONFIGS = [
    # SS-small
    dict(
        dataset_id="9n6mf",
        n=10,
        experiment_n_workers=-1,
        target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
        list_of_tsdr_options=get_tsifter_options(),
        list_of_diag_options=get_list_of_diag_options(),
        pair_of_use_manually_selected_metrics=[True],
        metric_types_pairs=[{
            "services": True,
            "containers": True,
            "middlewares": False,
            "nodes": False,
        }],
        progress=True,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    ),
    # SS-medium
    dict(
        dataset_id="9n6mf",
        n=10,
        experiment_n_workers=-1,
        target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
        list_of_tsdr_options=get_tsifter_options(),
        list_of_diag_options=get_list_of_diag_options(mode="fast-only"),
        pair_of_use_manually_selected_metrics=[False],
        metric_types_pairs=[{
            "services": True,
            "containers": True,
            "middlewares": False,
            "nodes": False,
        }],
        progress=True,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    ),
    # SS-large
    dict(
        dataset_id="9n6mf",
        n=10,
        experiment_n_workers=-1,
        target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
        list_of_tsdr_options=get_tsifter_options(),
        list_of_diag_options=get_list_of_diag_options(mode="fast-only"),
        pair_of_use_manually_selected_metrics=[False],
        metric_types_pairs=[{
            "services": True,
            "containers": True,
            "middlewares": True,
            "nodes": False,
        }],
        progress=True,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    ),

    # TT-small
    dict(
        dataset_id="m9dgg",
        n=10,
        experiment_n_workers=-1,
        target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
        list_of_tsdr_options=get_tsifter_options(),
        list_of_diag_options=get_list_of_diag_options(mode="fast-only"),
        pair_of_use_manually_selected_metrics=[True],
        metric_types_pairs=[{
            "services": True,
            "containers": True,
            "middlewares": False,
            "nodes": False,
        }],
        progress=True,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    ),
    # TT-medium
    dict(
        dataset_id="m9dgg",
        n=10,
        experiment_n_workers=-1,
        target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
        list_of_tsdr_options=get_tsifter_options(),
        list_of_diag_options=get_list_of_diag_options(mode="fast-only"),
        pair_of_use_manually_selected_metrics=[False],
        metric_types_pairs=[{
            "services": True,
            "containers": True,
            "middlewares": False,
            "nodes": False,
        }],
        progress=True,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    ),
    # TT-large
    dict(
        dataset_id="m9dgg",
        n=10,
        experiment_n_workers=-1,
        target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
        list_of_tsdr_options=get_tsifter_options(),
        list_of_diag_options=get_list_of_diag_options(mode="fast-only"),
        pair_of_use_manually_selected_metrics=[False],
        metric_types_pairs=[{
            "services": True,
            "containers": True,
            "middlewares": True,
            "nodes": False,
        }],
        progress=True,
        timeout_sec=DEFAULT_TIMEOUT_SEC * 3,
    ),
]
