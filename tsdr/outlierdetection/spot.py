import numpy as np
from ads_evt import biSPOT

DEFAULT_PROBA = 1e-4
DEFAULT_N_POINTS = 10


def spot(
    train_y: np.ndarray, test_y: np.ndarray, proba: float = DEFAULT_PROBA, n_points: int = DEFAULT_N_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    model = biSPOT(q=proba, n_points=n_points)
    model.fit(init_data=train_y, data=test_y)
    model.initialize()
    results = model.run(with_alarm=True)
    scores: list[float] = []
    for index, (upper, lower) in enumerate(zip(results["upper_thresholds"], results["lower_thresholds"])):
        width: float = upper - lower
        if width <= 0:
            width = 1
        if test_y[index] > upper:
            scores.append((test_y[index] - upper) / width)
        elif test_y[index] < lower:
            scores.append((lower - test_y[index]) / width)
        else:
            scores.append(0)

    return np.array(scores), np.array(results["alarms"])


def detect_anomalies_with_spot(
    x: np.ndarray, anomalous_start_idx: int, proba: float = DEFAULT_PROBA, n_points: int = DEFAULT_N_POINTS
) -> tuple[bool, float]:
    test_start_idx = x.shape[0] - (anomalous_start_idx + 1)
    train, test = x[:test_start_idx], x[test_start_idx:]
    scores, alarms = spot(train, test, proba=proba, n_points=n_points)
    return alarms.size > 0, np.max(scores)
