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
            rcd_n_iters=100,
            rcd_topk=5,
            rcd_n_workers=1,
            rcd_n_workers_seed_ensamble=1,
            sli_anomaly_start_time_index=sli_anomaly_start_time_index,
        ),
        # dict(  # CG+RW-2
        #     use_pyrca=True,
        #     enable_prior_knowledge=True,
        #     method="call_graph",
        #     walk_method="rw-2",
        #     sli_anomaly_start_time_index=sli_anomaly_start_time_index,
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
        # comment out the following becuase of error of cyclic graph
        # dict(  # CG+HT
        #     use_pyrca=True,
        #     enable_prior_knowledge=True,
        #     method="call_graph",
        #     walk_method="ht",
        #     sli_anomaly_start_time_index=sli_anomaly_start_time_index,
        #     pyrca_boundary_index=pyrca_boundary_index,
        # ),
    ],
    # from_orig=(True, 180, 20),
    pair_of_use_manually_selected_metrics=[False],
    progress=True,
    timeout_sec=60 * 60 * 1,
)
