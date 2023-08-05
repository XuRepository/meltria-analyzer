from diagnoser.daggnn import config

ANOMALOUS_LOOKBACK_WINDOW = 20 * 4  # 20 minutes


CONFIG = dict(
    dataset_id="9n6mf",
    n=10,
    experiment_n_workers=-1,
    target_chaos_types={"pod-cpu-hog", "pod-memory-hog", "pod-network-latency"},
    list_of_tsdr_options=[
        dict(  # TSifter
            enable_unireducer=False,
            enable_multireducer=True,
            step2_clustering_method_name="changepoint-kde",
            step2_changepoint_multi_change_points=True,
            step2_changepoint_n_bkps=1,
            step2_changepoint_penalty="bic",
            step2_changepoint_kde_bandwidth="silverman",
            step2_changepoint_representative_method=False,
            step2_changepoint_cost_model="l2",
            step2_clustering_n_workers=1,
        ),
    ],
    # 1. e-Diagnosis
    # 2. RCD
    # 3. CausalRCA+PageRank
    # 4. PC+RW-2
    # 5. GES+RW-2
    # 6. LiNGAM+RW-2
    # 7. PC+PageRank
    # 8. GES+PageRank
    # 9. LiNGAM+PageRank
    list_of_diag_options=[
        dict(  # e-Diagnosis
            use_pyrca=True,
            root_metric_type="latency",
            disable_orientation=True,
            method="epsilon_diagnosis",
            pyrca_boundary_index=100,
        ),
        dict(  # LiNGAM+HT
            use_pyrca=True,
            root_metric_type="latency",
            disable_orientation=True,
            method="lingam",
            walk_method="ht",
            pyrca_boundary_index=100,
        ),
        dict(  # LiNGAM+RW-2
            use_pyrca=True,
            root_metric_type="latency",
            disable_orientation=True,
            method="lingam",
            walk_method="rw-2",
            pyrca_boundary_index=100,
        ),
    ],
    pair_of_use_manually_selected_metrics=[False],
    progress=True,
)
