import gc
import itertools
import logging
import time
from typing import Any

import joblib
import numpy.typing as npt
import pandas as pd

from simulation.synthetic_data import (DATA_DIR, generate_synthetic_data,
                                       load_data)
from tsdr.birch import detect_anomalies_with_birch
from tsdr.clustering.pearsonr import pearsonr_as_dist
from tsdr.clustering.sbd import sbd
from tsdr.multireducer import (
    change_point_clustering_with_kde,
    change_point_clustering_with_kde_by_changepoints, dbscan_clustering)
from tsdr.unireducer import (changepoint_model, fluxinfer_model,
                             two_samp_test_model, zscore_nsigma_model)

REDUCTION_METHODS = ["MetricSifter", "NSigma", "BIRCH", "K-S test", "FluxInfer-AD", "HDBS-SBD", "HDBS-R", "None"]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)


def reduce_features(
    method: str,
    normal_data_df: pd.DataFrame, abnormal_data_df: pd.DataFrame, true_root_causes: list[str], graph: pd.DataFrame, anomaly_propagated_nodes: set[str],
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    concated_data_df = pd.concat([normal_data_df, abnormal_data_df], axis=0, ignore_index=True)

    sta: float = time.perf_counter()

    remained_metrics = []
    lookback_window_size = 4 * 20  # 20min
    match method:
        case "None":
            remained_metrics = concated_data_df.columns.tolist()
        case "Ideal":
            remained_metrics = list(anomaly_propagated_nodes)
        case "MetricSifter w/o segmentation":
            for col in concated_data_df.columns:
                result = changepoint_model(
                    concated_data_df[col].values,
                    step1_method_name="changepoint",
                    step1_changepoint_search_method="pelt",
                    step1_changepoint_cost_model="l2",
                    step1_changepoint_penalty="bic",
                )
                if result.has_kept:
                    remained_metrics.append(col)
        case "MetricSifter":
            cinfo, remove_metrics = change_point_clustering_with_kde(
                concated_data_df,
                search_method="pelt",
                cost_model="l2",
                penalty="bic",
                n_bkps=1,
                kde_bandwidth=kwargs.get("metricsifter_bandwidth", 2.50),
                kde_bandwidth_adjust=1.0,
                multi_change_points=True,
                representative_method=False,
                segment_selection_method="weighted_max",
                n_jobs=1,
            )
            remained_metrics = list(cinfo.keys())
        case "NSigma":
            for col in concated_data_df.columns:
                result = zscore_nsigma_model(
                    concated_data_df[col].values,
                    step1_zscore_nsigma_anomalous_start_idx=lookback_window_size,
                    step1_zscore_nsigma_n_sigmas=3,
                    step1_zscore_nsigma_robust=False,
                )
                if result.has_kept:
                    remained_metrics.append(col)
        case "BIRCH":
            birch_res = detect_anomalies_with_birch(
                concated_data_df,
                step1_birch_anomalous_start_idx=-lookback_window_size,
            )
            remained_metrics = [
                metric
                for metric, is_normal in birch_res.items()
                if not is_normal
            ]
        case "K-S test":
            for col in concated_data_df.columns:
                result = two_samp_test_model(
                    concated_data_df[col].values,
                    step1_two_samp_test_method="ks",
                    step1_two_samp_test_alpha=0.05,
                    step1_two_samp_test_seg_idx=-lookback_window_size,
                )
                if result.has_kept:
                    remained_metrics.append(col)
        case "FluxInfer-AD":
            for col in concated_data_df.columns:
                result = fluxinfer_model(
                    concated_data_df[col].values,
                    step1_fluxinfer_sigma_threshold=3.0,
                )
                if result.has_kept:
                    remained_metrics.append(col)
        case "HDBS-SBD":
            cinfo, remove_metrics = dbscan_clustering(
                concated_data_df, dist_func=sbd, eps=0, min_pts=1, algorithm="hdbscan")
            remained_metrics = list(cinfo.keys())
        case "HDBS-R":
            cinfo, remove_metrics = dbscan_clustering(
                concated_data_df, dist_func=pearsonr_as_dist, eps=0, min_pts=1, algorithm="hdbscan")
            remained_metrics = list(cinfo.keys())

    end: float = time.perf_counter()
    elapsed: float = end - sta

    total_metrics = concated_data_df.columns.tolist()
    true_removed_metrics = list(set(total_metrics) - anomaly_propagated_nodes)

    remove_metrics = list(set(total_metrics) - set(remained_metrics))
    normal_data_df.drop(columns=remove_metrics, inplace=True)
    abnormal_data_df.drop(columns=remove_metrics, inplace=True)
    graph.drop(columns=remove_metrics, index=remove_metrics, inplace=True)

    root_cause_recall = len(set(remained_metrics) & set(true_root_causes)) / len(true_root_causes)
    should_interacted_nodes = anomaly_propagated_nodes & set(remained_metrics)
    recall = (len(should_interacted_nodes) / len(anomaly_propagated_nodes)) if len(anomaly_propagated_nodes) > 0 else 0.0
    precision = (len(should_interacted_nodes) / len(remained_metrics)) if len(remained_metrics) > 0 else 0.0
    if (recall + precision) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * recall * precision / (recall + precision)
    specificity = len(set(remove_metrics) & set(true_removed_metrics)) / len(true_removed_metrics) if len(remained_metrics) > 0 else 0.0
    bacc = (recall + specificity) / 2
    stats = {
        "reduction_method": method,
        "num_remained": len(remained_metrics),
        "num_total": len(concated_data_df.columns),
        "reduction_rate": len(remove_metrics) / len(concated_data_df.columns),
        "root_cause_recall": root_cause_recall,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1_score": f1_score,
        "bacc": bacc,
        "elapsed_time_tsdr": elapsed,
    }
    return normal_data_df, abnormal_data_df, graph, stats


def sweep_feature_reduction(dataset) -> list[dict]:
    def _reduce(
        method, normal_data_df, abnormal_data_df, true_root_causes, graph, anomaly_propagated_nodes, **data_params,
    ):
        logger.info(f"Running {method} on anomaly type {data_params['anomaly_type']} with data scale {data_params}, func_type: {data_params['func_type']}, noise_type: {data_params['noise_type']}, weight_generator:{data_params['weight_generator']}, trian_no {data_params['trial_no']}...")

        normal_data_df, abnormal_data_df, graph, stats = reduce_features(method, normal_data_df, abnormal_data_df, true_root_causes, graph, anomaly_propagated_nodes)

        return stats | data_params

    results = joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(_reduce)(
            fl_method, normal_data_df, abnormal_data_df, true_root_causes, graph, anomaly_propagated_nodes,
            **({"trial_no": trial_no, "anomaly_type": anomaly_type, "func_type": func_type, "noise_type": noise_type, "weight_generator": weight_generator} | data_scale_param),
        )
        for (anomaly_type, data_scale_param, func_type, noise_type, weight_generator, trial_no), (normal_data_df, abnormal_data_df, true_root_causes, graph, anomaly_propagated_nodes) in dataset
        for fl_method in REDUCTION_METHODS
    )
    assert results is not None
    return results


def sweep_feature_reduction_with_generation(
    anomaly_types: list[int],
    data_scale_params: list[dict[str, int]],
    func_types: list[str],
    noise_types: list[str],
    weight_generators: list[str],
    trial_nos: list[int],
    n_jobs: int = -1,
):
    def _load_and_reduce(anomaly_type, data_params, func_type, noise_type, weight_generator, trial_no):
        logger.info(f"Running anomaly type {anomaly_type} with data scale {data_params}, func_type: {func_type}, noise_type: {noise_type}, weight_generator:{weight_generator}, trian_no {trial_no}...")

        normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes = generate_synthetic_data(
            num_node=data_params["num_node"],
            num_edge=data_params["num_edge"],
            num_normal_samples=data_params["num_normal_samples"],
            num_abnormal_samples=data_params["num_abnormal_samples"],
            anomaly_type=anomaly_type,
            func_type=func_type,
            noise_type=noise_type,
            weight_generator=weight_generator,
        )

        stats = []
        for fl_method in REDUCTION_METHODS:
            _normal_df, _abnormal_df, _, stat = reduce_features(
                fl_method, normal_data_df.copy(), abnormal_data_df.copy(), true_root_causes, adjacency_df.copy(), anomaly_propagated_nodes,
            )
            stats.append(
                stat | data_params | {
                    "trial_no": trial_no, "anomaly_type": anomaly_type, "func_type": func_type, "noise_type": noise_type, "weight_generator": weight_generator,
                },
            )
            del _normal_df, _abnormal_df
            gc.collect()

        del normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes
        gc.collect()

        return stats

    params = list(itertools.product(
        anomaly_types, data_scale_params, func_types, noise_types, weight_generators, trial_nos,
    ))
    results = joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
        joblib.delayed(_load_and_reduce)(*param) for param in params
    )
    assert results is not None
    return sum(results, [])


def sweep_load_and_reduction(
    anomaly_types: list[int],
    data_scale_params: list[dict[str, int]],
    func_types: list[str],
    noise_types: list[str],
    weight_generators: list[str],
    trial_nos: list[int],
    methods: list[str] = REDUCTION_METHODS,
    n_jobs: int = -1,
):
    def _load_and_reduce(anomaly_type, data_params, func_type, noise_type, weight_generator, trial_no):
        logger.info(f"Running anomaly type {anomaly_type} with data scale {data_params}, func_type: {func_type}, noise_type: {noise_type}, weight_generator:{weight_generator}, trian_no {trial_no}...")

        normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes = load_data(
            anomaly_type=anomaly_type,
            data_params=dict(
                num_node=data_params["num_node"],
                num_edge=data_params["num_edge"],
                num_normal_samples=data_params["num_normal_samples"],
                num_abnormal_samples=data_params["num_abnormal_samples"],
            ),
            func_type=func_type,
            noise_type=noise_type,
            weight_generator=weight_generator,
            trial_no=trial_no,
        )

        stats = []
        for fl_method in methods:
            _normal_df, _abnormal_df, graph, stat = reduce_features(
                fl_method, normal_data_df.copy(), abnormal_data_df.copy(), true_root_causes, adjacency_df.copy(), anomaly_propagated_nodes)
            stats.append(
                stat | data_params | {
                    "trial_no": trial_no, "anomaly_type": anomaly_type, "func_type": func_type, "noise_type": noise_type, "weight_generator": weight_generator,
                },
            )

            # save
            data_id = joblib.hash([anomaly_type, data_params, func_type, noise_type, weight_generator, trial_no], hash_name="sha1")
            assert data_id is not None
            (DATA_DIR / data_id / fl_method).mkdir(parents=True, exist_ok=True)

            for obj, name in (
                (_normal_df, "normal_df"),
                (_abnormal_df, "abnormal_df"),
                (stat, "stat"),
            ):
                joblib.dump(obj, DATA_DIR / data_id / fl_method / f"{name}.pkl.bz2", compress=("bz2", 3))

            del _normal_df, _abnormal_df, graph
            gc.collect()

        del normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes
        gc.collect()

        return stats

    params = list(itertools.product(
        anomaly_types, data_scale_params, func_types, noise_types, weight_generators, trial_nos,
    ))
    results = joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
        joblib.delayed(_load_and_reduce)(*param) for param in params
    )
    assert results is not None
    return sum(results, [])


def sweep_metricsifter_params(
    bandwidths: list[float],
    anomaly_types: list[int],
    data_scale_params: list[dict[str, int]],
    func_types: list[str],
    noise_types: list[str],
    weight_generators: list[str],
    trial_nos: list[int],
    n_jobs: int = -1,
):

    def _load_and_reduce(metricsifter_bandwidth, anomaly_type, data_params, func_type, noise_type, weight_generator, trial_no):
        logger.info(f"Running anomaly type {anomaly_type} with data scale {data_params}, func_type: {func_type}, noise_type: {noise_type}, weight_generator:{weight_generator}, trian_no {trial_no}...")

        normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes = load_data(
            anomaly_type=anomaly_type,
            data_params=dict(
                num_node=data_params["num_node"],
                num_edge=data_params["num_edge"],
                num_normal_samples=data_params["num_normal_samples"],
                num_abnormal_samples=data_params["num_abnormal_samples"],
            ),
            func_type=func_type,
            noise_type=noise_type,
            weight_generator=weight_generator,
            trial_no=trial_no,
        )

        _, _, _, stat = reduce_features(
            "MetricSifter", normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes,
            metricsifter_bandwidth=metricsifter_bandwidth,
        )
        return stat | data_params | {
            "trial_no": trial_no, "anomaly_type": anomaly_type, "func_type": func_type, "noise_type": noise_type, "weight_generator": weight_generator, "metricsifter_bandwidth": metricsifter_bandwidth,
        }

    params = list(itertools.product(
        bandwidths, anomaly_types, data_scale_params, func_types, noise_types, weight_generators, trial_nos,
    ))
    results = joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
        joblib.delayed(_load_and_reduce)(*param) for param in params
    )
    assert results is not None
    return results
