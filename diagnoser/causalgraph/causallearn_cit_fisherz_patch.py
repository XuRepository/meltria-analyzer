from math import log, sqrt

import numpy as np
from causallearn.utils.cit import FisherZ
from scipy.stats import chi2, norm


def register_cls_method(cls, method_name: str, method) -> None:
    setattr(cls, method_name, lambda *args, **kwargs: method(*args, **kwargs))


def fisherz_test(self, X, Y, condition_set=None):
    """
    Perform an independence test using Fisher-Z's test.

    Parameters
    ----------
    X, Y and condition_set : column indices of data

    Returns
    -------
    p : the p-value of the test
    """
    Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
    if cache_key in self.pvalue_cache:
        return self.pvalue_cache[cache_key]
    var = Xs + Ys + condition_set
    sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
    try:
        inv = np.linalg.pinv(sub_corr_matrix)
        # inv = np.linalg.inv(sub_corr_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Data correlation matrix is singular. Cannot run fisherz test. Please check your data.")
    # TODO: use np.sqrt instead of sqrt to avoid 'math domain error'
    r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
    if r >= 1.0:
        r = 1.0 - 1e-15
    elif r <= -1.0:
        r = -1.0 + 1e-15
    Z = 0.5 * np.log((1 + r) / (1 - r))
    X = np.sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
    p = 2 * (1 - norm.cdf(abs(X)))
    self.pvalue_cache[cache_key] = p
    return p


register_cls_method(FisherZ, "__call__", fisherz_test)
