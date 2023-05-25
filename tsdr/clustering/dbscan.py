import warnings
from collections.abc import Callable

import hdbscan
import numpy as np
import sklearn.cluster
from sklearn.exceptions import EfficiencyWarning
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph, sort_graph_by_row_values


def learn_clusters(
    X: np.ndarray,
    dist_func: Callable,
    min_pts: int = 1,
    algorithm: str = "dbscan",
    eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= 2:
        # avoid "ValueError: Expected n_neighbors <= n_samples,  but n_samples = 2, n_neighbors = 3"
        return np.array([]), np.array([])

    match algorithm:
        case "dbscan":
            dist_graph = kneighbors_graph(X=X, n_neighbors=min_pts, mode="distance", metric=dist_func)
            sort_graph_by_row_values(dist_graph, warn_when_not_sorted=False)

            distances = dist_graph.toarray().flatten()
            assert len(distances) != 0, f"distances is empty: {distances}, {X}"
            assert np.isnan(distances).sum() == 0, f"distances has NaN: {distances}, {X}"
            if eps == 0.0:
                eps = np.nanmax(distances) / 4  # see DBSherlock paper
                if eps == 0.0:  # avoid "ValueError: eps=0.0 is invalid: must be greater than or equal to 1e-20."
                    eps = 1e-20

            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=EfficiencyWarning)
                labels = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_pts, metric="precomputed").fit_predict(
                    dist_graph
                )
            return labels, dist_graph.toarray()
        case "hdbscan":
            distance_matrix = pairwise_distances(X, metric=dist_func, force_all_finite=False)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                metric="precomputed",
                allow_single_cluster=True,
                core_dist_n_jobs=1,
            ).fit(distance_matrix)
            return clusterer.labels_, distance_matrix
        case _:
            raise ValueError(f"Unknown algorithm: {algorithm}")
