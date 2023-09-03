from eval.config.common.tsifter_best_params import TSIFTER_BEST_PARAM

CONFIG = dict(
    dataset_id="m9dgg",
    list_of_tsdr_options=[TSIFTER_BEST_PARAM],
    use_manually_selected_metrics=[True, False],
    progress=True,
    max_chaos_case_num=3,
)
