from typing import Any

from eval.config.common.tsifter_best_params import TSIFTER_BEST_PARAM

ANOMALOUS_LOOKBACK_WINDOW = 20 * 4  # 20 minutes

TSDR_OPTIONS: list[dict[str, Any]] = [
    dict(  # None
        enable_unireducer=False,
        enable_multireducer=False,
    ),
    TSIFTER_BEST_PARAM,  # TSifter
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
        # step2_clustering_granularity="container",
    ),
    dict(  # HDBSCAN-PERSON
        enable_unireducer=False,
        enable_multireducer=True,
        step2_dbscan_min_pts=1,
        step2_dbscan_dist_type="pearsonr",
        step2_dbscan_algorithm="hdbscan",
        step2_clustering_series_type="raw",
        step2_clustering_choice_method="medoid",
        # step2_clustering_granularity="container",
    ),
]
