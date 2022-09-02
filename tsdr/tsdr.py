import random
import time
from concurrent import futures
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import hamming, pdist, squareform

import tsdr.unireducer as unireducer
from meltria.loader import count_metrics
from meltria.priorknowledge.priorknowledge import PriorKnowledge
from tsdr.clustering import dbscan
from tsdr.clustering.kshape import kshape
from tsdr.clustering.metricsnamecluster import cluster_words
from tsdr.clustering.sbd import sbd, silhouette_score
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule
from tsdr.unireducer import UnivariateSeriesReductionResult, has_variation

ELAPSED_TIME_NUM_DECIMAL_PLACES: int = 4


class Tsdr:
    def __init__(
        self,
        univariate_series_func_or_name: Callable[[np.ndarray, Any], UnivariateSeriesReductionResult] | str,
        **kwargs
    ) -> None:
        self.params = kwargs
        if callable(univariate_series_func_or_name):
            setattr(self, 'univariate_series_func', univariate_series_func_or_name)
        elif type(univariate_series_func_or_name) == str:
            func: Callable = unireducer.map_model_name_to_func(univariate_series_func_or_name)
            setattr(self, 'univariate_series_func', func)
        else:
            raise TypeError(f'Invalid type of step1 mode: {type(univariate_series_func_or_name)}')

    def univariate_series_func(self, series: np.ndarray, **kwargs: Any) -> UnivariateSeriesReductionResult:
        return unireducer.ar_based_ad_model(series, **kwargs)

    def filter_out_no_change_metrics(self, series: pd.DataFrame) -> pd.DataFrame:
        def filter(x) -> bool:
            if np.all(np.isnan(x)):
                return False
            diff_x = np.diff(x)
            return not (np.all(x == x[0]) or np.all(diff_x == diff_x[0]))
        return series.loc[:, series.apply(filter)]

    def detect_failure_start_point(self, sli: np.ndarray, sigma_threshold=3) -> tuple[int, float]:
        """ Detect failure start point in SLO metrics.
        The method uses outliter detection with 'robust z-score' and 3-sigma rule.
        """
        fi_time = self.params['time_fault_inject_time_index']
        train, test = np.split(sli, [fi_time])
        coeff = scipy.stats.norm.ppf(0.75)-scipy.stats.norm.ppf(0.25)
        iqr = np.quantile(train, 0.75) - np.quantile(train, 0.25)
        niqr = iqr / coeff
        median = np.median(train)
        for i, v in enumerate(test):
            if np.abs((v - median)/niqr) > sigma_threshold:
                return (fi_time+i, v)
        return (0, 0.0)

    def reduce_by_failure_detection_time(
        self,
        series: pd.DataFrame,
        results: dict[str, UnivariateSeriesReductionResult],
        sli: np.ndarray,
        sigma_threshold: int = 3,
    ) -> pd.DataFrame:
        """ reduce series by failure detection time
        """
        outliers = detect_with_n_sigma_rule(
            x=sli,
            test_start_time=self.params['time_fault_inject_time_index'],
            sigma_threshold=sigma_threshold,
        )
        failure_detection_time: int = 0 if outliers.size == 0 else outliers[0]

        kept_series_labels: list[str] = []
        for resuts_col, res in results.items():
            if not res.has_kept:
                continue
            change_start_time = res.change_start_point[0]
            if failure_detection_time == 0 or failure_detection_time >= change_start_time:
                kept_series_labels.append(resuts_col)
        return series[kept_series_labels]

    def run(
        self, X: pd.DataFrame, pk: PriorKnowledge, max_workers: int,
    ) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame, float]], dict[str, Any], dict[str, np.ndarray]]:
        stat: list[tuple[pd.DataFrame, pd.DataFrame, float]] = []
        stat.append((X, count_metrics(X), 0.0))

        # step0
        start: float = time.time()

        series: pd.DataFrame = self.filter_out_no_change_metrics(X)

        elapsed_time: float = round(time.time() - start, ELAPSED_TIME_NUM_DECIMAL_PLACES)
        stat.append((series, count_metrics(series), elapsed_time))

        # step1
        start = time.time()

        reduced_series1, step1_results, anomaly_points = self.reduce_univariate_series(series, max_workers)

        elapsed_time = round(time.time() - start, ELAPSED_TIME_NUM_DECIMAL_PLACES)
        stat.append((reduced_series1, count_metrics(reduced_series1), elapsed_time))

        if self.params['step1_residual_integral_change_start_point']:
            # step1.5
            start = time.time()

            sli_name: str = pk.get_root_metrics()[0]  # TODO: choose SLI metrics formally
            reduced_series1 = self.reduce_by_failure_detection_time(
                reduced_series1, step1_results, series[sli_name].to_numpy(),
                sigma_threshold=self.params['step1_residual_integral_change_start_point_n_sigma'],
            )

            elapsed_time = round(time.time() - start, ELAPSED_TIME_NUM_DECIMAL_PLACES)
            stat.append((reduced_series1, count_metrics(reduced_series1), elapsed_time))

        # step2
        start = time.time()

        match series_type := self.params['step2_clustering_series_type']:
            case 'raw':
                df_before_clustering = reduced_series1.apply(scipy.stats.zscore)
            case 'anomaly_score', 'binary_anomaly_score':
                tmp_dict_to_df: dict[str, np.ndarray] = {}
                for name, res in step1_results.items():
                    if res.has_kept:
                        if series_type == 'anomaly_score':
                            tmp_dict_to_df[name] = scipy.stats.zscore(res.anomaly_scores)
                        elif series_type == 'binary_anomaly_score':
                            tmp_dict_to_df[name] = res.binary_scores()
                df_before_clustering = pd.DataFrame(tmp_dict_to_df)
            case _:
                raise ValueError(f'step2_clustered_series_type is invalid {series_type}')

        reduced_series2, clustering_info = self.reduce_multivariate_series(
            df_before_clustering.copy(), pk, max_workers,
        )

        elapsed_time = round(time.time() - start, ELAPSED_TIME_NUM_DECIMAL_PLACES)
        stat.append((reduced_series2, count_metrics(reduced_series2), elapsed_time))

        return stat, clustering_info, anomaly_points

    def reduce_univariate_series(
        self,
        useries: pd.DataFrame,
        n_workers: int,
    ) -> tuple[pd.DataFrame, dict[str, UnivariateSeriesReductionResult], dict[str, np.ndarray]]:
        results: dict[str, UnivariateSeriesReductionResult] = {}
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_col = {}
            for col in useries.columns:
                series: np.ndarray = useries[col].to_numpy()
                future = executor.submit(self.univariate_series_func, series, **self.params)
                future_to_col[future] = col
            reduced_cols: list[str] = []
            for future in futures.as_completed(future_to_col):
                col = future_to_col[future]
                result: UnivariateSeriesReductionResult = future.result()
                results[col] = result
                if result.has_kept:
                    reduced_cols.append(col)
        anomaly_points = {col: res.outliers for col, res in results.items()}
        return useries[reduced_cols], results, anomaly_points

    def reduce_multivariate_series(
        self,
        series: pd.DataFrame,
        pk: PriorKnowledge,
        n_workers: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        def make_clusters(
            df: pd.DataFrame,
            **kwargs: Any,
        ) -> futures.Future:
            method_name = kwargs['step2_clustering_method_name']
            choice_method = kwargs['step2_clustering_choice_method']

            future: futures.Future

            if method_name == 'hierarchy':
                dist_type = kwargs['step2_hierarchy_dist_type']
                dist_threshold = kwargs['step2_hierarchy_dist_threshold']
                linkage_method = kwargs['step2_hierarchy_linkage_method']

                if dist_type == 'sbd':
                    future = executor.submit(
                        hierarchical_clustering,
                        df, sbd, dist_threshold, choice_method, linkage_method,
                    )
                elif dist_type == 'hamming':
                    if dist_threshold >= 1.0:
                        # make the distance threshold intuitive
                        dist_threshold /= series.shape[0]
                    future = executor.submit(
                        hierarchical_clustering,
                        df, hamming, dist_threshold, choice_method, linkage_method,
                    )
                else:
                    raise ValueError('dist_func must be "sbd" or "hamming"')
            elif method_name == 'dbscan':
                future = executor.submit(
                    dbscan_clustering,
                    df, kwargs['step2_dbscan_dist_type'], kwargs['step2_dbscan_min_pts'], choice_method,
                )
            else:
                raise ValueError('method_name must be "hierarchy" or "dbscan"')
            return future

        clustering_info: dict[str, Any] = {}
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Clustering metrics by service including services, containers and middlewares metrics
            # TODO: node metrics clustering
            future_list: list[futures.Future] = []
            for service, containers in pk.get_containers_of_service().items():
                # 1. service-level clustering
                # TODO: retrieve service metrics efficently
                service_metrics_df = series.loc[:, series.columns.str.startswith(f"s-{service}_")]
                if len(service_metrics_df.columns) > 1:
                    future_list.append(
                        make_clusters(service_metrics_df, **self.params),
                    )
                # 2. container-level clustering
                for container in containers:
                    # perform clustering in each type of metric
                    # TODO: retrieve container and middleware metrics efficently
                    container_metrics_df = series.loc[
                        :, series.columns.str.startswith((f"c-{container}_", f"m-{container}_"))]
                    if len(container_metrics_df.columns) <= 1:
                        continue
                    future_list.append(
                        make_clusters(container_metrics_df, **self.params),
                    )
            # 3. node-level clustering
            for node in pk.get_nodes():
                node_metrics_df = series.loc[:, series.columns.str.startswith(f"n-{node}_")]
                if len(node_metrics_df.columns) <= 1:
                    continue
                future_list.append(
                    make_clusters(node_metrics_df, **self.params),
                )
            for future in futures.as_completed(future_list):
                c_info, remove_list = future.result()
                clustering_info.update(c_info)
                series.drop(remove_list, axis=1, inplace=True)
        return series, clustering_info


def reduce_series_with_cv(data_df: pd.DataFrame, cv_threshold: float = 0.002):
    reduced_by_cv_df = pd.DataFrame()
    for col in data_df.columns:
        data = data_df[col].values
        if has_variation(data, cv_threshold):
            reduced_by_cv_df[col] = data_df[col]
    return reduced_by_cv_df


def hierarchical_clustering(
    target_df: pd.DataFrame,
    dist_func: Callable,
    dist_threshold: float,
    choice_method: str = 'medoid',
    linkage_method: str = 'single',
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

    if choice_method == 'medoid':
        return choose_metric_with_medoid(target_df.columns, cluster_dict, dist_matrix)
    elif choice_method == 'maxsum':
        return choose_metric_with_maxsum(target_df, cluster_dict)
    else:
        raise ValueError('choice_method is required.')


def dbscan_clustering(
    target_df: pd.DataFrame,
    dist_func: str,
    min_pts: int,
    choice_method: str = 'medoid',
) -> tuple[dict[str, Any], list[str]]:
    labels, dist_matrix = dbscan.learn_clusters(target_df.values.T, dist_func, min_pts)

    cluster_dict: dict[int, list[int]] = {}
    for i, v in enumerate(labels):
        if v in cluster_dict:
            cluster_dict[v].append(i)
        else:
            cluster_dict[v] = [i]

    if choice_method == 'medoid':
        return choose_metric_with_medoid(target_df.columns, cluster_dict, dist_matrix)
    elif choice_method == 'maxsum':
        return choose_metric_with_maxsum(target_df, cluster_dict)
    else:
        raise ValueError('choice_method is required.')


def choose_metric_with_medoid(
    columns: pd.Index,
    cluster_dict: dict[int, list[int]],
    dist_matrix: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    clustering_info, remove_list = {}, []
    for c in cluster_dict:
        cluster_metrics = cluster_dict[c]
        if len(cluster_metrics) == 1:
            continue
        if len(cluster_metrics) == 2:
            # Select the representative metric at random
            shuffle_list = random.sample(cluster_metrics, len(cluster_metrics))
            clustering_info[columns[shuffle_list[0]]] = [
                columns[shuffle_list[1]]]
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
    """ Choose metrics which has max of sum of datapoints in each metrics in each cluster. """
    clustering_info, remove_list = {}, []
    for c in cluster_dict:
        cluster_metrics: list[int] = cluster_dict[c]
        if len(cluster_metrics) == 1:
            continue
        if len(cluster_metrics) > 1:
            cluster_columns = data_df.columns[cluster_metrics]
            series_with_sum: pd.Series = data_df[cluster_columns].sum(numeric_only=True)
            label_with_max: str = series_with_sum.idxmax()
            sub_metrics: list[str] = list(series_with_sum.loc[series_with_sum.index != label_with_max].index)
            clustering_info[label_with_max] = sub_metrics
            remove_list += sub_metrics
    return clustering_info, remove_list


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
    cluster_metrics: list[str], columns: dict[str, Any], centroid: int,
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
            clustering_info[columns[representative_metric]].append(
                columns[r])
    return clustering_info, remove_list


def kshape_clustering(
    target_df: pd.DataFrame, service_name: str, executor,
) -> tuple[dict[str, Any], list[str]]:
    future_list = []

    data: np.ndarray = target_df.apply(scipy.stats.zscore).values.T
    for n in np.arange(2, data.shape[0]):
        future_list.append(
            executor.submit(
                create_clusters, data, target_df.columns, service_name, n,
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
            executor.submit(select_representative_metric, data,
                            cluster_metrics, target_df.columns, centroid[c])
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


def sieve_reduce_series(data_df):
    return reduce_series_with_cv(data_df)


def sieve_clustering(reduced_df, services_list, max_workers):
    clustering_info = {}

    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Clustering metrics by services including services, containers and middlewares
        for ser in services_list:
            target_df = reduced_df.loc[:, reduced_df.columns.str.startswith(
                ("s-{}_".format(ser), "c-{}_".format(ser), "c-{}-".format(ser), "m-{}_".format(ser), "m-{}-".format(ser)))]
            if len(target_df.columns) in [0, 1]:
                continue
            c_info, remove_list = kshape_clustering(target_df, ser, executor)
            clustering_info.update(c_info)
            reduced_df = reduced_df.drop(remove_list, axis=1)

    return reduced_df, clustering_info
