import random
from collections import defaultdict
from collections.abc import Callable
from concurrent import futures
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.signal
import scipy.stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from tsdr.clustering import changepoint, dbscan
from tsdr.clustering.kshape import kshape
from tsdr.clustering.metricsnamecluster import cluster_words
from tsdr.clustering.pearsonr import pearsonr
from tsdr.clustering.sbd import sbd, silhouette_score


def hierarchical_clustering(
    target_df: pd.DataFrame,
    dist_func: Callable | str,
    dist_threshold: float,
    choice_method: str = "medoid",
    linkage_method: str = "single",
    sli_data: np.ndarray | None = None,
    top_k: int = 1,
) -> tuple[dict[str, Any], list[str]]:
    dist = pdist(target_df.values.T, metric=dist_func)
    dist_matrix: np.ndarray = squareform(dist)
    z: np.ndarray = linkage(dist, method=linkage_method, metric=dist_func)
    labels: np.ndarray = fcluster(z, t=dist_threshold, criterion="distance")
    cluster_dict: dict[int, list[int]] = {}
    for i, v in enumerate(labels):
        if v in cluster_dict:
            cluster_dict[v].append(i)
        else:
            cluster_dict[v] = [i]

    match choice_method:
        case "medoid":
            return choose_metric_with_medoid(
                target_df.columns, cluster_dict, dist_matrix
            )
        case "maxsum":
            return choose_metric_with_maxsum(target_df, cluster_dict)
        case "max_cluster":
            return choose_metrics_with_max_cluster(
                target_df.columns.to_list(), cluster_dict
            )
        case "cluster_of_max_corr_to_sli":
            assert sli_data is not None
            return choose_metrics_within_cluster_of_max_corr_to_sli(
                target_df, cluster_dict, dist_matrix, sli_data, top_k=top_k
            )
        case _:
            raise ValueError("choice_method is required.")


def dbscan_clustering(
    target_df: pd.DataFrame,
    dist_func: str | Callable,
    min_pts: int,
    algorithm: str,
    eps: float,
    choice_method: str = "medoid",
    sli_data: np.ndarray | None = None,
    top_k: int = 1,
) -> tuple[dict[str, Any], list[str]]:
    labels, dist_matrix = dbscan.learn_clusters(
        X=target_df.values.T,
        dist_func=dist_func,
        min_pts=min_pts,
        algorithm=algorithm,
        eps=eps,
    )

    cluster_dict: dict[int, list[int]] = {}
    for i, v in enumerate(labels):
        if v in cluster_dict:
            cluster_dict[v].append(i)
        else:
            cluster_dict[v] = [i]

    match choice_method:
        case "medoid":
            return choose_metric_with_medoid(
                target_df.columns, cluster_dict, dist_matrix
            )
        case "maxsum":
            return choose_metric_with_maxsum(target_df, cluster_dict)
        case "max_cluster":
            return choose_metrics_with_max_cluster(
                target_df.columns.to_list(), cluster_dict
            )
        case "cluster_of_max_corr_to_sli":
            assert sli_data is not None
            return choose_metrics_within_cluster_of_max_corr_to_sli(
                target_df, cluster_dict, dist_matrix, sli_data, top_k=top_k
            )
        case _:
            raise ValueError("choice_method is required.")


def choose_metric_with_medoid(
    columns: pd.Index,
    cluster_dict: dict[int, list[int]],
    dist_matrix: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    clustering_info, remove_list = {}, []
    for c in cluster_dict:
        cluster_metrics = cluster_dict[c]
        if len(cluster_metrics) == 1:
            clustering_info[columns[cluster_metrics[0]]] = []
        elif len(cluster_metrics) == 2:
            # Select the representative metric at random
            shuffle_list = random.sample(cluster_metrics, len(cluster_metrics))
            clustering_info[columns[shuffle_list[0]]] = [columns[shuffle_list[1]]]
            remove_list.append(columns[shuffle_list[1]])
        elif len(cluster_metrics) > 2:
            # Select medoid as the representative metric
            distances = []
            for met1 in cluster_metrics:
                dist_sum = 0
                for met2 in cluster_metrics:
                    if met1 == met2:
                        continue
                    dist_sum += dist_matrix[met1][met2]
                distances.append(dist_sum)
            medoid = cluster_metrics[np.argmin(distances)]
            clustering_info[columns[medoid]] = []
            for r in cluster_metrics:
                if r == medoid:
                    continue
                remove_list.append(columns[r])
                clustering_info[columns[medoid]].append(columns[r])
    return clustering_info, remove_list


def choose_metric_with_maxsum(
    data_df: pd.DataFrame,
    cluster_dict: dict[int, list[int]],
) -> tuple[dict[str, Any], list[str]]:
    """Choose metrics which has max of sum of datapoints in each metrics in each cluster."""
    clustering_info, remove_list = {}, []
    for c in cluster_dict:
        cluster_metrics: list[int] = cluster_dict[c]
        if len(cluster_metrics) == 1:
            # TODO
            continue
        elif len(cluster_metrics) > 1:
            cluster_columns = data_df.columns[cluster_metrics]
            series_with_sum: pd.Series = data_df[cluster_columns].sum(numeric_only=True)
            label_with_max: str = series_with_sum.idxmax()
            sub_metrics: list[str] = list(
                series_with_sum.loc[series_with_sum.index != label_with_max].index
            )
            clustering_info[label_with_max] = sub_metrics
            remove_list += sub_metrics
    return clustering_info, remove_list


def choose_metrics_with_max_cluster(
    columns: list[str],
    cluster_dict: dict[int, list[int]],
) -> tuple[dict[str, Any], list[str]]:
    """Choose metrics which has max of number of datapoints in each metrics in each cluster."""
    if len(cluster_dict) == 0:
        return {}, []
    max_cluster_key = max(cluster_dict, key=cluster_dict.get)
    max_cluster_metrics: list[int] = cluster_dict[max_cluster_key]
    clustering_info: dict[str, list[str]] = {
        columns[i]: [] for i in max_cluster_metrics
    }
    remove_list: list[str] = [
        col for i, col in enumerate(columns) if i not in max_cluster_metrics
    ]
    return clustering_info, remove_list


def choose_metrics_within_cluster_of_max_corr_to_sli(
    data_df: pd.DataFrame,
    cluster_dict: dict[int, list[int]],
    dist_matrix: np.ndarray,
    sli_data: np.ndarray,
    top_k: int = 1,
) -> tuple[dict[str, Any], list[str]]:
    medoid_cinfo = choose_metric_with_medoid(
        data_df.columns, cluster_dict, dist_matrix
    )[0]
    coefs: dict[str, float] = {}
    for rep_metric in medoid_cinfo.keys():
        coef = pearsonr(data_df[rep_metric].to_numpy(), sli_data, apply_abs=True)
        coefs[rep_metric] = coef

    clustering_info: dict[str, list[str]] = {}
    for rep_metric, _ in sorted(coefs.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]:
        clustering_info[rep_metric] = medoid_cinfo[rep_metric]

    keep_list: set[str] = set(clustering_info.keys()) | set(
        m for sub_metrics in clustering_info.values() for m in sub_metrics
    )
    remove_list = list(set(data_df.columns) - keep_list)
    return clustering_info, remove_list


def change_point_clustering(
    data: pd.DataFrame,
    cost_model: str,
    n_bkps: int,
    proba_threshold: float,
    choice_method: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    cluster_allow_single_cluster: bool,
    sli_data: np.ndarray | None = None,
    n_jobs: int = -1,
) -> tuple[dict[str, Any], list[str]]:
    assert n_bkps >= 1, f"n_bkps must be >= 1: {n_bkps}"
    assert (
        0.0 <= proba_threshold and proba_threshold <= 1.0
    ), f"proba_threshold must be in [0.0, 1.0]: {proba_threshold}"

    change_points: list[int] = changepoint.detect_changepoints(data, cost_model, n_bkps, n_jobs)
    metrics: list[str] = data.columns.tolist()
    clusters: changepoint.Clusters = changepoint.cluster_changepoints(
        change_points=change_points,
        metrics=metrics,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_allow_single_cluster=cluster_allow_single_cluster,
    )
    assert len(clusters) > 0, f"clusters is empty: {clusters}"

    clusters = clusters.inliner(
        threshold=proba_threshold,
        eps=int(cluster_selection_epsilon),
    )

    # return all metrics if all clusters are noise
    if all([c.is_noise() for c in clusters]):
        return {metric: [] for metric in metrics}, []

    choiced_cluster = changepoint.choice_cluster(
        clusters, choice_method=choice_method, sli_data=sli_data
    )

    keep_metrics: set[str] = set([m.metric_name for m in choiced_cluster.members])
    remove_metrics: list[str] = list(set(metrics) - keep_metrics)
    clustering_info: dict[str, list[str]] = {metric: [] for metric in keep_metrics}
    return clustering_info, remove_metrics


def change_point_clustering_with_kde_by_changepoints(
    data: pd.DataFrame,
    changepoints: list[npt.ArrayLike],
    kde_bandwidth: float | str,
    kde_bandwidth_adjust: float,
) -> tuple[dict[str, Any], list[str]]:
    metrics: list[str] = data.columns.tolist()
    cluster_label_to_metrics, _ = changepoint.cluster_multi_changepoints(
        multi_change_points=[[cp[0] for cp in cps] for cps in changepoints],
        metrics=metrics,
        time_series_length=data.shape[0],
        kde_bandwidth=kde_bandwidth,
        kde_bandwidth_adjust=kde_bandwidth_adjust,
    )
    if cluster_label_to_metrics:
        choiced_cluster = max(cluster_label_to_metrics.items(), key=lambda x: len(x[1]))
        keep_metrics: set[str] = set(choiced_cluster[1])
    else:
        keep_metrics = set()
    remove_metrics: list[str] = list(set(metrics) - keep_metrics)
    clustering_info: dict[str, list[str]] = {metric: [] for metric in keep_metrics}
    return clustering_info, remove_metrics


def change_point_clustering_with_kde(
    data: pd.DataFrame,
    search_method: str,
    cost_model: str,
    penalty: str | float,  # penalty is not used if step2_changepoint_multi_change_points is false.
    kde_bandwidth: float | str,
    kde_bandwidth_adjust: float,
    n_bkps: int = 1,  # n_bkps is not used if multi_change_points is true.
    multi_change_points: bool = False,
    representative_method: bool = False,
    n_jobs: int = -1,
) -> tuple[dict[str, Any], list[str]]:
    if representative_method:
        assert multi_change_points, "multi_change_points must be true if representative_method is true."

    metrics: list[str] = data.columns.tolist()
    if not multi_change_points:
        change_points: list[int] = changepoint.detect_changepoints(data, cost_model, n_bkps, n_jobs)
        cluster_labels, _ = changepoint.cluster_changepoints_with_kde(
            change_points=change_points,
            time_series_length=data.shape[0],
            kde_bandwidth=kde_bandwidth,
            kde_bandwidth_adjust=kde_bandwidth_adjust,
        )
        cluster_label_to_metrics = defaultdict(set)
        for label, metric in zip(cluster_labels, metrics):
            if label == changepoint.NO_CHANGE_POINTS:  # skip no anomaly metrics
                continue
            cluster_label_to_metrics[label].add(metric)
    else:
        if representative_method:
            cluster_label_to_metrics = changepoint.cluster_multi_changepoints_by_repsentative_change_point(
                X=data,
                cost_model=cost_model,
                penalty=penalty,
                kde_bandwidth=kde_bandwidth,
                kde_bandwidth_adjust=kde_bandwidth_adjust,
                n_jobs=n_jobs,
            )
        else:
            changepoints = changepoint.detect_multi_changepoints(
                X=data, search_method=search_method, cost_model=cost_model, penalty=penalty, n_jobs=n_jobs
            )
            cluster_label_to_metrics, _ = changepoint.cluster_multi_changepoints(
                multi_change_points=changepoints,
                metrics=metrics,
                time_series_length=data.shape[0],
                kde_bandwidth=kde_bandwidth,
                kde_bandwidth_adjust=kde_bandwidth_adjust,
            )

    if cluster_label_to_metrics:
        choiced_cluster = max(cluster_label_to_metrics.items(), key=lambda x: len(x[1]))
        keep_metrics: set[str] = set(choiced_cluster[1])
    else:
        keep_metrics = set()
    remove_metrics: list[str] = list(set(metrics) - keep_metrics)
    clustering_info: dict[str, list[str]] = {metric: [] for metric in keep_metrics}
    return clustering_info, remove_metrics


def create_clusters(data: pd.DataFrame, columns: list[str], service_name: str, n: int):
    words_list: list[str] = [col[2:] for col in columns]
    init_labels = cluster_words(words_list, service_name, n)
    results = kshape(data, n, initial_clustering=init_labels)
    label = [0] * data.shape[0]
    cluster_center = []
    cluster_num = 0
    for res in results:
        if not res[1]:
            continue
        for i in res[1]:
            label[i] = cluster_num
        cluster_center.append(res[0])
        cluster_num += 1
    if len(set(label)) == 1:
        return None
    return (label, silhouette_score(data, label), cluster_center)


def select_representative_metric(
    data: pd.DataFrame,
    cluster_metrics: list[str],
    columns: dict[str, Any],
    centroid: int,
) -> tuple[dict[str, Any], list[str]]:
    clustering_info: dict[str, Any] = {}
    remove_list: list[str] = []
    if len(cluster_metrics) == 1:
        return clustering_info, remove_list
    if len(cluster_metrics) == 2:
        # Select the representative metric at random
        shuffle_list: list[str] = random.sample(cluster_metrics, len(cluster_metrics))
        clustering_info[columns[shuffle_list[0]]] = [columns[shuffle_list[1]]]
        remove_list.append(columns[shuffle_list[1]])
    elif len(cluster_metrics) > 2:
        # Select the representative metric based on the distance from the centroid
        distances = []
        for met in cluster_metrics:
            distances.append(sbd(centroid, data[met]))
        representative_metric = cluster_metrics[np.argmin(distances)]
        clustering_info[columns[representative_metric]] = []
        for r in cluster_metrics:
            if r == representative_metric:
                continue
            remove_list.append(columns[r])
            clustering_info[columns[representative_metric]].append(columns[r])
    return clustering_info, remove_list


def kshape_clustering(
    target_df: pd.DataFrame,
    service_name: str,
    executor,
) -> tuple[dict[str, Any], list[str]]:
    future_list = []

    data: np.ndarray = target_df.apply(scipy.stats.zscore).values.T
    for n in np.arange(2, data.shape[0]):
        future_list.append(
            executor.submit(
                create_clusters,
                data,
                target_df.columns,
                service_name,
                n,
            )
        )
    labels, scores, centroids = [], [], []
    for future in futures.as_completed(future_list):
        cluster = future.result()
        if cluster is None:
            continue
        labels.append(cluster[0])
        scores.append(cluster[1])
        centroids.append(cluster[2])

    idx = np.argmax(scores)
    label = labels[idx]
    centroid = centroids[idx]
    cluster_dict = {}
    for i, v in enumerate(label):
        if v not in cluster_dict:
            cluster_dict[v] = [i]
        else:
            cluster_dict[v].append(i)

    future_list = []
    for c, cluster_metrics in cluster_dict.items():
        future_list.append(
            executor.submit(
                select_representative_metric,
                data,
                cluster_metrics,
                target_df.columns,
                centroid[c],
            )
        )
    clustering_info = {}
    remove_list = []
    for future in futures.as_completed(future_list):
        c_info, r_list = future.result()
        if c_info is None:
            continue
        clustering_info.update(c_info)
        remove_list.extend(r_list)

    return clustering_info, remove_list
