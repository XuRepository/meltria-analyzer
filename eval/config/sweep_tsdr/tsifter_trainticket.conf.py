from eval.config.common.tsdr_options import get_tsifter_options

CONFIG = dict(
    dataset_id="m9dgg",
    list_of_tsdr_options=get_tsifter_options(),
    use_manually_selected_metrics=[True, False],
    progress=True,
    max_chaos_case_num=5,
)

