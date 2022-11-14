from collections.abc import Callable
from typing import Any

import hdbscan
import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.cluster
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from tsdr.clustering.pearsonr import pearsonr_as_dist


def learn_clusters(
    X: np.ndarray,
    dist_func: str | Callable,
    min_pts: int = 1,
    algorithm: str = "dbscan",
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= 2:
        # avoid "ValueError: Expected n_neighbors <= n_samples,  but n_samples = 2, n_neighbors = 3"
        return np.array([]), np.array([])

    match algorithm:
        case "dbscan":
            nn_fit = NearestNeighbors(n_neighbors=min_pts, metric=dist_func).fit(X)
            distances = nn_fit.kneighbors()[0]
            dist_square_matrix: scipy.sparse.csr_matrix = nn_fit.radius_neighbors_graph(
                mode="distance", sort_results=True
            )

            eps = max(distances.flatten()) / 4  # see DBSherlock paper

            labels = sklearn.cluster.DBSCAN(
                eps=eps,
                min_samples=min_pts,
                metric="precomputed",
            ).fit_predict(dist_square_matrix)
            return labels, dist_square_matrix.toarray()
        case "hdbscan":
            distance_matrix = pairwise_distances(X, metric=dist_func, force_all_finite=False)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                metric=dist_func,
                allow_single_cluster=True,
            ).fit(distance_matrix)
            return clusterer.labels_, distance_matrix
        case _:
            raise ValueError(f"Unknown algorithm: {algorithm}")
