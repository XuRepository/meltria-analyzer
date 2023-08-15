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
    # 1. e-Diagnosis
    # 2. RCD
    # 3. CausalRCA+PageRank
    # 4. PC+RW-2
    # 5. GES+RW-2
    # 6. LiNGAM+RW-2
    # 7. PC+PageRank
    # 8. GES+PageRank
    # 9. LiNGAM+PageRank
    # 10. PC+HT
    # 11. GES+HT
    # 12. LiNGAM+HT
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
        # dict(  # PC+RW-2
        #     use_pyrca=True,
        #     enable_prior_knowledge=True,
        #     method="pc",
        #     walk_method="rw-2",
        #     sli_anomaly_start_time_index=sli_anomaly_start_time_index,
        #     pyrca_boundary_index=pyrca_boundary_index,
        # ),
        # dict(  # GES+RW-2
        #     use_pyrca=True,
        #     enable_prior_knowledge=True,
        #     method="ges",
        #     walk_method="rw-2",
        #     sli_anomaly_start_time_index=sli_anomaly_start_time_index,
        #     pyrca_boundary_index=pyrca_boundary_index,
        # ),
        dict(  # CG+RW-2
            use_pyrca=True,
            enable_prior_knowledge=True,
            method="call_graph",
            walk_method="rw-2",
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
            pyrca_boundary_index=pyrca_boundary_index,
        ),
        dict(  # LiNGAM+RW-2
            use_pyrca=True,
            enable_prior_knowledge=True,
            method="lingam",
            walk_method="rw-2",
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
            pyrca_boundary_index=pyrca_boundary_index,
        ),
        # dict(  # PC+PageRank
        #     use_pyrca=True,
        #     enable_prior_knowledge=True,
        #     method="pc",
        #     walk_method="pagerank",
        #     pyrca_boundary_index=pyrca_boundary_index,
        # ),
        # dict(  # GES+PageRank
        #     use_pyrca=True,
        #     method="ges",
        #     walk_method="pagerank",
        #     pyrca_boundary_index=pyrca_boundary_index,
        # ),
        dict(  # CG+PageRank
            use_pyrca=True,
            enable_prior_knowledge=True,
            method="call_graph",
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
        # dict(  # PC+HT
        #     use_pyrca=True,
        #     enable_prior_knowledge=True,
        #     method="pc",
        #     walk_method="ht",
        #     sli_anomaly_start_time_index=sli_anomaly_start_time_index,
        #     pyrca_boundary_index=pyrca_boundary_index,
        # ),
        # dict(  # GES+HT
        #     use_pyrca=True,
        #     enable_prior_knowledge=True,
        #     method="ges",
        #     walk_method="ht",
        #     sli_anomaly_start_time_index=sli_anomaly_start_time_index,
        #     pyrca_boundary_index=pyrca_boundary_index,
        # ),
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
            config.Config().to_prefixed_dict("causalrca"), use_causalrca=True
        ),
    ],
    # from_orig=(True, 180),
    pair_of_use_manually_selected_metrics=[True, False],
    progress=True,
    timeout_sec=60 * 60 * 5,
)
