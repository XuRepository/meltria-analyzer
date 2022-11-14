from typing import Any

import numpy as np
import scipy.stats


def pearsonr_as_dist(X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> float:
    r = scipy.stats.pearsonr(X, Y)[0]
    return 1 - abs(r) if r is not np.nan else 0.0
