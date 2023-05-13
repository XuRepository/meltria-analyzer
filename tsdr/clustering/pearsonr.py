from typing import Any

import numpy as np
import scipy.stats


def pearsonr_as_dist(X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> float:
    r = scipy.stats.pearsonr(X, Y)[0]
    return 1 - abs(r) if r is not np.nan else 0.0


def pearsonr(X: np.ndarray, Y: np.ndarray, apply_abs: bool = False) -> float:
    r = scipy.stats.pearsonr(X, Y)[0]
    return abs(r) if apply_abs else r


def pearsonr_left_shift(x: np.ndarray, sli: np.ndarray, l_p: int, apply_abs: bool = False) -> float:
    """Pearsonr left time shift
        see CauseRank paper [Lu+, CCGrid2022].

    # - t_alert: the start time of the alert (in some unit of time)
    # - l_corr: the historical data length (in the same unit of time) used for correlation coefficient calculation
    # - l_test: the length of time (in the same unit of time) after the start of the alert to consider for correlation calculation
    - l_p: the maximum failure propagation time (in the same unit of time)
    """
    max_corr: float = -1.0  # initialize maximum correlation to negative value

    for i in range(l_p + 1):
        # calculate time periods for correlation coefficients
        # x_data = sli[t_alert - l_corr - i : t_alert + l_test - i]
        x_data = x[l_p - i : -i]
        # sli_data = sli[t_alert - l_corr : t_alert + l_test]
        sli_data = sli[l_p:]
        # calculate correlation coefficient between x and k
        corr: float = scipy.stats.pearsonr(x_data, sli_data)[0]
        # update maximum correlation if new value is greater
        max_corr = max(max_corr, abs(corr) if apply_abs else corr)
    # return final correlation value
    return max_corr
