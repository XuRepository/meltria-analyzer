import numpy as np
import pandas as pd
import scipy.stats


def detect_with_n_sigma_rule(
    x: np.ndarray | pd.Series,
    test_start_time: int,
    sigma_threshold: int = 3,
    robust: bool = True,
) -> np.ndarray:
    """Detect outliers 'robust z-score' and n-sigma rule.
    deprecated
    """
    train, test = np.split(x, [test_start_time])
    if robust:
        coeff = scipy.stats.norm.ppf(0.75) - scipy.stats.norm.ppf(0.25)
        iqr = np.quantile(train, 0.75) - np.quantile(train, 0.25)
        niqr = iqr / coeff
        median = np.median(train)
        outlier_idx = np.where(np.abs(test - median) > niqr * sigma_threshold)[0] + test_start_time
        return outlier_idx
    else:
        mean, std = np.mean(train), np.std(train)
        outlier_idx = np.where(np.abs(test - mean) > std * sigma_threshold)[0] + test_start_time
        return outlier_idx


def zscore_nsigma(train: np.ndarray, test: np.ndarray, n_sigmas: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    mu, sigma = np.mean(train), np.std(train)
    if sigma == 0.0:
        sigma = 1.0
    scores = np.abs((test - mu) / sigma)
    anomalies = test[scores > n_sigmas]
    return anomalies, scores


COEFF = scipy.stats.norm.ppf(0.75) - scipy.stats.norm.ppf(0.25)


def robust_zscore_nsigma(
    train: np.ndarray, test: np.ndarray, test_start_time: int, n_sigmas: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    iqr = np.quantile(train, 0.75) - np.quantile(train, 0.25)
    niqr = iqr / COEFF
    median = np.median(train)
    scores = np.abs(test - median)
    anomalies = test[scores > niqr * n_sigmas]
    return anomalies, scores


def detect_anomalies_with_zscore_nsigma(
    x: np.ndarray,
    anomalous_start_idx: int,
    n_sigmas: float = 3.0,
    robust: bool = False,
) -> tuple[bool, float]:
    test_start_idx = x.shape[0] - (anomalous_start_idx + 1)
    train, test = x[:test_start_idx], x[test_start_idx:]
    if robust:
        alarms, scores = robust_zscore_nsigma(train, test, test_start_idx, n_sigmas)
    else:
        alarms, scores = zscore_nsigma(train, test, n_sigmas)
    return alarms.size > 0, scores.max()
