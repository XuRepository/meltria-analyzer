from eval.config.common.tsifter_params_sensitivity import TSDR_OPTIONS

CONFIG = dict(
    dataset_id="9n6mf",
    list_of_tsdr_options=TSDR_OPTIONS,
    use_manually_selected_metrics=[True],
    metric_types_pairs=[
    {
        "services": True,
        "containers": True,
        "middlewares": False,
        "nodes": False,
    }],
    progress=True,
    max_chaos_case_num=5,
)
