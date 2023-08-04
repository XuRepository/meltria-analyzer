ANOMALOUS_LOOKBACK_WINDOW = 20 * 4  # 20 minutes

CONFIG = dict(
    dataset_id="9n6mf",
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
        dict(  # NSigma
            enable_unireducer=True,
            enable_multireduce=False,
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
    use_manually_selected_metrics=[False],
    progress=True,
)
