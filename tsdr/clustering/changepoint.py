from __future__ import annotations

import warnings
from collections import UserList, defaultdict
from dataclasses import dataclass
from typing import Final

import hdbscan
import numpy as np
import pandas as pd
import ruptures as rpt
import scipy.stats
from joblib import Parallel, delayed
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity

NO_CHANGE_POINTS: Final[int] = -1

@dataclass
class ClusterMember:
    metric_name: str
    change_point: int
    proba: float


@dataclass
class Cluster:
    cluster_id: int  # -1 means a noise cluster
    centroid: int  # -1 means a noise cluster
    members: list[ClusterMember]

    @classmethod
    def from_raw_members(
        cls, cluster_id: int, centroid: int, members: list[tuple[str, int, float]]
    ) -> Cluster:
        return cls(cluster_id, centroid, [ClusterMember(*member) for member in members])

    def _has_near_members(self, target: ClusterMember, eps: int) -> bool:
        _members = sorted(self.members, key=lambda m: m.change_point)
        for _member in _members:
            if _member.change_point == target.change_point:
                continue
            if abs(_member.change_point - target.change_point) < eps:
                return True
        return False

    def _inliers(self, threshold: float = 0.0, eps: int = 1) -> list[ClusterMember]:
        assert 0 <= threshold <= 1
        return [
            m
            for m in self.members
            if m.proba >= threshold or self._has_near_members(m, eps)
        ]

    def size(self) -> int:
        return len(self.members)

    def is_noise(self) -> bool:
        return self.cluster_id == -1


class Clusters(UserList[Cluster]):
    def __init__(self, clusters: list[Cluster]) -> None:
        super().__init__(UserList(clusters))

    @classmethod
    def from_raw_values(
        cls, clusterer: hdbscan.HDBSCAN, metrics: list[str], change_points: list[int]
    ) -> Clusters:
        _cluster_id_to_centroid = {
            cluster_id: clusterer.weighted_cluster_centroid(cluster_id)[0]
            for cluster_id in np.unique(clusterer.labels_)
            if cluster_id != -1  # skip noise cluster
        }
        _cluster_id_to_members = defaultdict(list)
        for label, metric, change_point, proba in zip(
            clusterer.labels_, metrics, change_points, clusterer.probabilities_
        ):
            centroid: int = _cluster_id_to_centroid.get(
                label, -1
            )  # -1 means a noise cluster
            _cluster_id_to_members[(label, centroid)].append(
                (metric, change_point, proba)
            )

        clusters: list[Cluster] = []
        for (cluster_id, centroid), members in sorted(
            _cluster_id_to_members.items(), key=lambda x: x[0][1]
        ):
            clusters.append(Cluster.from_raw_members(cluster_id, centroid, members))
        return Clusters(clusters)

    def inliner(
        self, threshold: float = 1.0, eps: int = 1,
    ) -> Clusters:
        _orig_clusters = self.data
        return Clusters(
            [
                Cluster(
                    c.cluster_id,
                    c.centroid,
                    c._inliers(threshold=threshold, eps=eps),
                )
                for c in _orig_clusters
                if c.cluster_id != -1
            ]
        )

    def cluster_of_max_size(self) -> Cluster:
        return max(self, key=lambda c: c.size())

    def noise(self) -> Cluster:
        return next(c for c in self.data if c.is_noise())


def detect_changepoints(
    data: pd.DataFrame, cost_model: str = "normal", n_bkps: int = 1, n_jobs: int = -1
) -> list[int]:
    metrics: list[str] = data.columns.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binseg = rpt.Binseg(model=cost_model, jump=1)

    def detect_changepoint(metric: str) -> int:
        return binseg.fit(data[metric].to_numpy()).predict(n_bkps=n_bkps)[0]

    change_points: list[int] | None = Parallel(n_jobs=n_jobs)(
        delayed(detect_changepoint)(metric) for metric in metrics
    )
    assert change_points is not None
    return change_points


def _detech_changepoints_with_missing_values(x: np.ndarray):
    """
    Detect changepoints with missing values
    Sample:
        input: [1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9, np.nan, np.nan])
        output: [2 6 9]
    """
    is_nan = np.isnan(x)
    # Get the index where NaN value changes
    nans = np.where(is_nan, 1, 0)
    change_indices = np.where(np.diff(nans) == 1)[0] + 1
    # If array starts with NaN, add 0 to indices
    if is_nan[0]:
        change_indices = np.concatenate(([0], change_indices))
    return change_indices


def detect_univariate_changepoints(x: np.ndarray, cost_model: str, penalty: str | float) -> list[int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algo = rpt.Binseg(model=cost_model, jump=1)
    n_segs = 2
    sigma = np.std(x)
    match penalty:
        case "aic":
            pen = n_segs * sigma * sigma
        case "bic":
            pen = np.log(x.size) * n_segs * sigma * sigma
        case _:
            pen = penalty
    cps = algo.fit(x).predict(pen=pen)[:-1]
    mvs = _detech_changepoints_with_missing_values(x)
    return sorted(list(set(cps) | set(mvs)))


def detect_multi_changepoints(
    X: pd.DataFrame,
    cost_model: str = "l2",
    penalty: str | float = "aic",
    n_jobs: int = -1,
) -> list[list[int]]:
    metrics: list[str] = X.columns.tolist()

    multi_change_points: list[list[int]] | None = Parallel(n_jobs=n_jobs)(
        delayed(detect_univariate_changepoints)(X[metric].to_numpy(), cost_model, penalty) for metric in metrics
    )
    assert multi_change_points is not None

    return multi_change_points


def cluster_multi_changepoints(
    multi_change_points: list[list[int]],
    metrics: list[str],
    time_series_length: int,
    kde_bandwidth: float | str = "silverman",
) -> tuple[dict[int, list[str]], dict[int, np.ndarray]]:
    cp_to_metrics: dict[int, list[str]] = defaultdict(list)
    for metric, change_points in zip(metrics, multi_change_points):
        if len(change_points) < 1:
            cp_to_metrics[NO_CHANGE_POINTS].append(metric)  # cp == -1 means no change point
            continue
        for cp in change_points:
            cp_to_metrics[cp].append(metric)

    flatten_change_points: list[int] = sum(multi_change_points, [])
    if len(flatten_change_points) == 0:
        return {}, {}

    _, label_to_change_points = cluster_changepoints_with_kde(
        flatten_change_points, time_series_length=time_series_length, kde_bandwidth=kde_bandwidth, unique_values=True,
    )

    label_to_metrics: dict[int, list[str]] = defaultdict(list)
    for label, cps in label_to_change_points.items():
        if label == NO_CHANGE_POINTS:  # skip no anomaly metrics
            continue
        for cp in cps:
            for metric in cp_to_metrics[cp]:
                label_to_metrics[label].append(metric)
    return label_to_metrics, label_to_change_points


def cluster_multi_changepoints_by_repsentative_change_point(
    X: pd.DataFrame,
    cost_model: str = "l2",
    penalty: str | float = "aic",
    kde_bandwidth: float | str = "silverman",
    n_jobs: int = -1,
) -> dict[int, list[str]]:
    multi_change_points = detect_multi_changepoints(
        X, cost_model=cost_model, penalty=penalty, n_jobs=n_jobs
    )
    metrics = X.columns.tolist()
    label_to_metrics, label_to_change_points = cluster_multi_changepoints(
        multi_change_points=multi_change_points,
        metrics=metrics,
        time_series_length=X.shape[0],
        kde_bandwidth=kde_bandwidth,
    )
    max_cluster = max(label_to_metrics.items(), key=lambda _: len(_[1]))
    rep_change_point = label_to_change_points[max_cluster[0]][0]  # choose first change point as representative in max cluster

    cp_to_metrics: dict[int, list[str]] = defaultdict(list)
    choiced_change_points = []
    for metric, cps in zip(metrics, multi_change_points):
        if len(cps) < 1:
            cp_to_metrics[NO_CHANGE_POINTS].append(metric)
            continue
        choiced_cp = cps[np.argmin(np.abs(cps - rep_change_point))]
        choiced_change_points.append(choiced_cp)
        cp_to_metrics[choiced_cp].append(metric)

    _, label_to_change_points = cluster_changepoints_with_kde(
        choiced_change_points, time_series_length=X.shape[0], kde_bandwidth=kde_bandwidth, unique_values=True,
    )

    label_to_metrics: dict[int, list[str]] = defaultdict(list)
    for label, cps in label_to_change_points.items():
        if label == NO_CHANGE_POINTS:  # skip no anomaly metrics
            continue
        for cp in cps:
            for metric in cp_to_metrics[cp]:
                label_to_metrics[label].append(metric)
    return label_to_metrics


def cluster_changepoints(
    change_points: list[int],
    metrics: list[str],
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    cluster_allow_single_cluster: bool,
) -> Clusters:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        metric=lambda x, y: abs(x - y),  # type: ignore
        allow_single_cluster=cluster_allow_single_cluster,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    ).fit(np.array([change_points]).T)
    clusters = Clusters.from_raw_values(
        clusterer, metrics=metrics, change_points=change_points
    )
    return clusters


def cluster_changepoints_with_kde(
    change_points: list[int],
    time_series_length: int,
    kde_bandwidth: float | str,
    unique_values: bool = False,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    assert len(change_points) > 0, "change_points should not be empty"

    x = np.array(change_points, dtype=int)
    kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(x.reshape(-1, 1))
    s = np.linspace(start=0, stop=time_series_length - 1, num=time_series_length // 2)  # If 'num' are larger, cluster size will be smaller.
    e = kde.score_samples(s.reshape(-1, 1))

    mi = argrelextrema(e, np.less)[0]
    clusters = []
    if len(mi) <= 0:
        clusters.append(np.arange(len(x)))
    else:
        clusters.append(np.where(x < s[mi][0])[0])  # most left cluster
        for i_cluster in range(len(mi) - 1):  # all middle cluster
            clusters.append(np.where((x >= s[mi][i_cluster]) * (x <= s[mi][i_cluster + 1]))[0])
        clusters.append(np.where(x >= s[mi][-1])[0])  # most right cluster

    labels = np.zeros(len(x), dtype=int)
    for label, x_args in enumerate(clusters):
        clusters[label] = np.unique(x[x_args]) if unique_values else x[x_args]
        labels[x_args] = label
    label_to_values: dict[int, np.ndarray] = {label: vals for label, vals in enumerate(clusters)}
    return labels, label_to_values


def choice_cluster(
    clusters: Clusters,
    choice_method: str,
    n_bkps: int = 1,
    sli_data: np.ndarray | None = None,
) -> Cluster:
    """Choose cluster including root cause metrics"""
    match choice_method:
        case "max_members_changepoint":
            return clusters.cluster_of_max_size()

        case "nearest_sli_changepoint":
            assert sli_data is not None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                binseg = rpt.Binseg(model="normal", jump=1)

            slis_cp: int = binseg.fit(scipy.stats.zscore(sli_data)).predict(
                n_bkps=n_bkps
            )[0]
            return min(
                clusters, key=lambda x: abs(x.centroid - slis_cp)
            )  # the distance between centroid and sli changepoint
        case _:
            raise ValueError("choice_method is required.")
