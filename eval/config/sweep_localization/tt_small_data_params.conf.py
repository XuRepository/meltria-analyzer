from eval.config.common.data_params import SAMPLING_SCALE_FACTORS, TIME_RANGES
from eval.config.common.tsifter_best_params import TSIFTER_BEST_PARAM
from eval.config.sweep_localization.common.localizations import \
    get_list_of_diag_options

CONFIG = dict(
    dataset_id="m9dgg",
    n=10,
    experiment_n_workers=-1,
    target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
    list_of_tsdr_options=[TSIFTER_BEST_PARAM],
    list_of_diag_options=get_list_of_diag_options(sampling_scale_factors=SAMPLING_SCALE_FACTORS),
    pair_of_use_manually_selected_metrics=[True],
    metric_types_pairs=[{
        "services": True,
        "containers": True,
        "middlewares": False,
        "nodes": False,
    }],
    sampling_scale_factors=SAMPLING_SCALE_FACTORS,
    time_ranges=TIME_RANGES,
    progress=True,
    timeout_sec=60 * 60 * 1,
    # from_orig=(True, 180, 20, 3),
)
