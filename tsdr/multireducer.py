import random
import warnings
from collections import defaultdict
from collections.abc import Callable
from concurrent import futures
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
import ruptures as rpt
import scipy.signal
import scipy.stats
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from tsdr.clustering import dbscan
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
    n_bkps: int,
    proba_threshold: float,
    choice_method: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    cluster_allow_single_cluster: bool,
    sli_data: np.ndarray | None = None,
    n_jobs=-1,
) -> tuple[dict[str, Any], list[str]]:
    assert n_bkps >= 1, f"n_bkps must be >= 1: {n_bkps}"
    assert (
        0.0 <= proba_threshold and proba_threshold <= 1.0
    ), f"proba_threshold must be in [0.0, 1.0]: {proba_threshold}"

    metrics: list[str] = data.columns.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binseg = rpt.Binseg(model="normal", jump=1)

    def detect_changepoint(metric: str) -> int:
        return binseg.fit(data[metric].to_numpy()).predict(n_bkps=n_bkps)[0]

    change_points: list[int] | None = Parallel(n_jobs=n_jobs)(
        delayed(detect_changepoint)(metric) for metric in metrics
    )
    assert change_points is not None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        metric="euclidean",
        allow_single_cluster=cluster_allow_single_cluster,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    ).fit(np.array([change_points]).T)

    # return all metrics if all labels (cluster_ids) are -1
    if np.all(clusterer.labels_ == -1):
        return {metric: [] for metric in metrics}, []

    cluster_id_to_centroid = {
        cluster_id: clusterer.weighted_cluster_centroid(cluster_id)[0]
        for cluster_id in np.unique(clusterer.labels_)
        if cluster_id != -1  # skip noise cluster
    }
    clusters_with_centroid: dict[tuple[int, int], list[str]] = defaultdict(list)
    for cluster_id, metric, change_point, prob in zip(
        clusterer.labels_, metrics, change_points, clusterer.probabilities_
    ):
        if cluster_id == -1 or prob <= proba_threshold:  # skip noise or outlier metric
            continue
        centroid = cluster_id_to_centroid[cluster_id]
        clusters_with_centroid[(cluster_id, centroid)].append(metric)
    assert (
        clusters_with_centroid
    ), f"clusters_with_centroid is empty: {clusters_with_centroid}, prob_threshold={proba_threshold}, (metric,cluster_id,probability)={list(zip(metrics, clusterer.labels_ ,clusterer.probabilities_))}"

    # Choose cluster including root cause metrics
    def choice_fn(x: dict[tuple[int, int], list[str]]):
        match choice_method:
            case "max_members_changepoint":
                return max(x.items(), key=lambda y: len(y[1]))
            case "nearest_sli_changepoint":
                assert sli_data is not None
                slis_cp: int = binseg.fit(scipy.stats.zscore(sli_data)).predict(
                    n_bkps=n_bkps
                )[0]
                return min(
                    x.items(), key=lambda y: abs(y[0][1] - slis_cp)
                )  # the distance between centroid and sli changepoint
            case _:
                raise ValueError("choice_method is required.")

    max_cluster = choice_fn(clusters_with_centroid)

    keep_metrics: set[str] = set(max_cluster[1])
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
