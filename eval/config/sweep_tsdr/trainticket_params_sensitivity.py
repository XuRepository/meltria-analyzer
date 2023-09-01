from eval.config.common.data_params import SAMPLING_SCALE_FACTORS, TIME_RANGES
from eval.config.common.tsifter_params_sensitivity import TSDR_OPTIONS

CONFIG = dict(
    dataset_id="m9dgg",
    list_of_tsdr_options=TSDR_OPTIONS,
    use_manually_selected_metrics=[True, False],
    progress=True,
    max_chaos_case_num=5,
    sampling_scale_factors=SAMPLING_SCALE_FACTORS,
    time_ranges=TIME_RANGES,
)
