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
        sigma = 0.0001
    scores = np.abs((test - mu) / sigma)
    anomalies = np.argwhere(scores > n_sigmas)
    return anomalies, scores


def zscore_nsigma_causerank(train: np.ndarray, test: np.ndarray, n_sigmas: float) -> tuple[bool, np.ndarray]:
    mu, sigma = np.mean(train), np.std(train)
    if sigma == 0.0:
        sigma = 0.0001
    scores = np.abs((test - mu) / sigma)
    s_x: float = np.max(scores, axis=0)
    alpha_x: float = 0 if s_x < n_sigmas else np.log1p(s_x)
    return alpha_x > 0, scores


COEFF = scipy.stats.norm.ppf(0.75) - scipy.stats.norm.ppf(0.25)


def robust_zscore_nsigma(train: np.ndarray, test: np.ndarray, n_sigmas: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    iqr = np.quantile(train, 0.75) - np.quantile(train, 0.25)
    niqr = iqr / COEFF
    median = np.median(train)
    scores = np.abs(test - median)
    anomalies = np.argwhere(scores > niqr * n_sigmas)
    return anomalies, scores


def robust_zscore_nsigma_causerank(train: np.ndarray, test: np.ndarray, n_sigmas: float) -> tuple[bool, np.ndarray]:
    iqr = np.quantile(train, 0.75) - np.quantile(train, 0.25)
    niqr = iqr / COEFF
    median = np.median(train)
    scores = np.abs(test - median)
    s_x: float = np.max(scores, axis=0)
    alpha_x: float = 0 if s_x < niqr * n_sigmas else np.log1p(s_x)
    return alpha_x > 0, scores



def detect_anomalies_with_zscore_nsigma(
    x: np.ndarray,
    anomalous_start_idx: int,
    n_sigmas: float = 3.0,
    robust: bool = False,
    return_score: bool = False,
) -> tuple[bool, float] | tuple[np.ndarray, np.ndarray]:
    test_start_idx = x.shape[0] - (anomalous_start_idx + 1)
    train, test = x[:test_start_idx], x[test_start_idx:]
    if robust:
        alarms, scores = robust_zscore_nsigma(train, test, n_sigmas)
    else:
        alarms, scores = zscore_nsigma(train, test, n_sigmas)
    return (alarms, scores) if return_score else (alarms.size > 0, scores.max())


def detect_anomalies_with_zscore_nsigma_causerank(
    x: np.ndarray,
    anomalous_start_idx: int,
    n_sigmas: float = 1.0,
    robust: bool = False,
) -> tuple[bool, np.ndarray]:
    test_start_idx = x.shape[0] - (anomalous_start_idx + 1)
    train, test = x[:test_start_idx], x[test_start_idx:]
    return robust_zscore_nsigma_causerank(train, test, n_sigmas) if robust else zscore_nsigma_causerank(train, test, n_sigmas)
