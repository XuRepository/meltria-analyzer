from eval.config.common.tsifter_ablation import TSDR_OPTIONS

CONFIG = dict(
    dataset_id="m9dgg",
    list_of_tsdr_options=TSDR_OPTIONS,
    use_manually_selected_metrics=[True, False],
    progress=True,
    max_chaos_case_num=3,
)
