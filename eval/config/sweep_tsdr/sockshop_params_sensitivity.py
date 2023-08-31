from eval.config.common.tsifter_params_sensitivity import TSDR_OPTIONS

CONFIG = dict(
    dataset_id="9n6mf",
    list_of_tsdr_options=TSDR_OPTIONS,
    use_manually_selected_metrics=[True, False],
    progress=True,
    max_chaos_case_num=5,
)
