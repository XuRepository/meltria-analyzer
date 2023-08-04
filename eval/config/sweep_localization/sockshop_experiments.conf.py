from diagnoser.daggnn import config

ANOMALOUS_LOOKBACK_WINDOW = 20 * 4  # 20 minutes


CONFIG = dict(
    dataset_id="9n6mf",
    n=10,
    experiment_n_workers=-1,
    target_chaos_types={"pod-cpu-hog", "pod-memory-hog", "pod-network-latency"},
    list_of_tsdr_options=[
        dict(  # None
            enable_unireducer=False,
            enable_multireducer=False,
        ),
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
        dict(  # NSigma
            enable_unireducer=True,
            enable_multireducer=False,
            step1_method_name="zscore_nsigma",
            step1_zscore_nsigma_anomalous_start_idx=ANOMALOUS_LOOKBACK_WINDOW,
            step1_zscore_nsigma_n_sigmas=3.0,
            step1_zscore_nsigma_robust=False,
        ),
        dict(  # Birch
            enable_unireducer=True,
            enable_multireducer=False,
            step1_method_name="birch_model",
            step1_birch_threshold=10,
            step1_birch_branching_factor=50,
            step1_birch_anomalous_start_idx=-ANOMALOUS_LOOKBACK_WINDOW,
        ),
        dict(  # K-S test
            enable_unireducer=True,
            enable_multireducer=False,
            step1_model_name="two_samp_test",
            step1_two_samp_test_method="ks",
            step1_two_samp_test_alpha=0.05,
            step1_two_samp_test_seg_idx=-ANOMALOUS_LOOKBACK_WINDOW  # 0 means division at midpoint
        ),
        dict(  # FluxInfer-AD
            enable_unireducer=True,
            enable_multireducer=False,
            step1_model_name="fluxinfer",
            step1_fluxinfer_sigma_threshold=3.0,
        ),
        dict(  # HDBSCAN-SBD
            enable_unireducer=False,
            enable_multireducer=True,
            step2_dbscan_min_pts=1,
            step2_dbscan_dist_type="sbd",
            step2_dbscan_algorithm="hdbscan",
            step2_clustering_series_type="raw",
            step2_clustering_choice_method="medoid",
        ),
        dict(  # HDBSCAN-PERSON
            enable_unireducer=False,
            enable_multireducer=True,
            step2_dbscan_min_pts=1,
            step2_dbscan_dist_type="pearsonr",
            step2_dbscan_algorithm="hdbscan",
            step2_clustering_series_type="raw",
            step2_clustering_choice_method="medoid",
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
        dict(  # LiNGAM+PageRank
            root_metric_type="throughput",
            enable_prior_knowledge=True,
            pc_library="lingam",
            disable_orientation=True,
            walk_method="pagerank",
        ),
        dict(  # RCD
            use_rcd=True,
            rcd_boundary_index=160,
            rcd_localized=True,
            rcd_gamma=5,
            rcd_bins=5,
            rcd_n_iters=10,
            rcd_topk=5,
            rcd_n_workers=1,
            rcd_n_workers_seed_ensamble=-1,
        ),
        dict(  # CausalRCA
            config.Config().to_prefixed_dict("causalrca"), use_causalrca=True
        ),
    ],
    from_orig=(True, 180),
    pair_of_use_manually_selected_metrics=[False],
    progress=True,
)
