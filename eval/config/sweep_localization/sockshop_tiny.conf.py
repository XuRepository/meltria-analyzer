from diagnoser.daggnn import config
from eval.config.common.tsdr_options import TSDR_OPTIONS

pyrca_boundary_index = 100
sli_anomaly_start_time_index = 160


CONFIG = dict(
    dataset_id="9n6mf",
    n=10,
    experiment_n_workers=-1,
    target_chaos_types={"pod-cpu-hog", "pod-memory-hog"},  # "pod-network-latency"},
    list_of_tsdr_options=TSDR_OPTIONS,
    list_of_diag_options=[
        dict(  # e-Diagnosis
            use_pyrca=True,
            method="epsilon_diagnosis",
            pyrca_boundary_index=pyrca_boundary_index,
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
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
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
        ),
        dict(  # CG+PageRank
            use_pyrca=True,
            enable_prior_knowledge=True,
            method="call_graph",
            walk_method="pagerank",
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
            pyrca_boundary_index=pyrca_boundary_index,
        ),
        dict(  # PC+PageRank
            use_pyrca=True,
            enable_prior_knowledge=True,
            method="pc",
            walk_method="pagerank",
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
            pyrca_boundary_index=pyrca_boundary_index,
        ),
        dict(  # LiNGAM+PageRank
            use_pyrca=True,
            enable_prior_knowledge=True,
            method="lingam",
            walk_method="pagerank",
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
            pyrca_boundary_index=pyrca_boundary_index,
        ),
        dict(  # PC+HT
            use_pyrca=True,
            enable_prior_knowledge=True,
            method="pc",
            walk_method="ht",
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
            pyrca_boundary_index=pyrca_boundary_index,
        ),
        dict(  # CG+HT
            use_pyrca=True,
            enable_prior_knowledge=True,
            method="lingam",
            walk_method="ht",
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
            pyrca_boundary_index=pyrca_boundary_index,
        ),
        dict(  # LiNGAM+HT
            use_pyrca=True,
            enable_prior_knowledge=True,
            method="lingam",
            walk_method="ht",
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
            pyrca_boundary_index=pyrca_boundary_index,
        ),
        dict(  # CausalRCA
            config.Config().to_prefixed_dict("causalrca"), use_causalrca=True,
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
        ),
    ],
    pair_of_use_manually_selected_metrics=[True],
    progress=True,
    timeout_sec=60 * 60 * 5,
)
