from collections.abc import Callable

import hdbscan
import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.cluster
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def learn_clusters(
    X: np.ndarray,
    dist_func: Callable,
    min_pts: int = 1,
    algorithm: str = "dbscan",
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= 2:
        # avoid "ValueError: Expected n_neighbors <= n_samples,  but n_samples = 2, n_neighbors = 3"
        return np.array([]), np.array([])

    match algorithm:
        case "dbscan":
            nn_fit = NearestNeighbors(n_neighbors=min_pts, metric=dist_func).fit(X)
            distances = nn_fit.kneighbors()[0].flatten()
            dist_square_matrix: scipy.sparse.csr_matrix = nn_fit.radius_neighbors_graph(
                mode="distance", sort_results=True
            )
            assert len(distances) != 0, f"distances is empty: {distances}, {X}"
            assert np.isnan(distances).sum() == 0, f"distances has NaN: {distances}, {X}"

            eps = np.nanmax(distances) / 4  # see DBSherlock paper
            if eps == 0.0:
                # avoid "ValueError: eps=0.0 is invalid: must be greater than or equal to 1e-20."
                eps = 1e-20

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
