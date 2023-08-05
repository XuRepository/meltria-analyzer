from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import Birch


class BirchAD:
    """Anomaly Detector with Birch Clustering
    see paper [Gulenko+,CLOUD2018] "Detecting Anomalous Behavior of Black-Box Services Modeled with Distance-Based Online Clustering"
    """

    def __init__(self, threshold=0.5, branching_factor=50) -> None:
        self.birch = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)

    def train(self, data) -> None:
        self.birch.fit(data)
        self.centroids_ = self.birch.subcluster_centers_
        self.radii_ = self.collect_radius()
        assert len(self.centroids_) == len(
            self.radii_
        ), f"len(self.centroids_)={len(self.centroids_)} != len(self.radii_)={len(self.radii_)}"

    def _get_leaves(self) -> list:
        leaf_ptr = self.birch.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    def collect_radius(self) -> np.ndarray:
        radii = []
        for leave in self._get_leaves():
            for subcluster in leave.subclusters_:
                radii.append(subcluster.radius)
        return np.array(radii)

    def is_normal(self, x: np.ndarray) -> bool:
        distances = np.array([np.linalg.norm(c - x) for c in self.centroids_])
        return bool(np.any(distances <= self.radii_))


def detect_anomalies_with_birch(data: pd.DataFrame, **kwargs: Any) -> dict[str, bool]:
    anomalous_start_idx: int = kwargs["step1_birch_anomalous_start_idx"]
    threshold: float = kwargs.get("step1_birch_threshold", 10.0)
    branching_factor: int = kwargs.get("step1_birch_branching_factor", 50)

    anomalous_data = data.iloc[anomalous_start_idx:, :]
    normal_data = data.iloc[anomalous_start_idx - anomalous_data.shape[0] : anomalous_start_idx, :]
    normal_mu, normal_sigma = normal_data.mean(), normal_data.std()

    def _zscore(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        if sigma == 0.0:
            sigma = 1
        return (x - mu) / sigma

    normal_data = normal_data.apply(lambda x: _zscore(x, normal_mu[x.name], normal_sigma[x.name]), axis=0)

    adtector = BirchAD(threshold=threshold, branching_factor=branching_factor)
    adtector.train(normal_data.values.T)

    results: dict[str, bool] = {}
    for col in anomalous_data.columns:
        x = _zscore(anomalous_data[col].values, normal_mu[col], normal_sigma[col])
        results[col] = adtector.is_normal(x)
    return results
