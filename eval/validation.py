import math
from typing import Any

import numpy as np
import pandas as pd
from ads_evt import biSPOT
import joblib

from eval.groundtruth import check_route, select_ground_truth_metrics_in_routes, check_cause_metrics
from meltria.loader import DatasetRecord
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule


def spot(train_y: np.ndarray, test_y: np.ndarray, proba: float = 1e-4, n_points: int = 10) -> tuple[np.ndarray, np.ndarray]:
    model = biSPOT(q=proba, n_points=n_points)
    model.fit(init_data=train_y, data=test_y)
    model.initialize()
    results = model.run(with_alarm=True)
    scores: list[float] = []
    for index, (upper, lower) in enumerate(
        zip(results["upper_thresholds"], results["lower_thresholds"])
    ):
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


def detect_anomalies_with_spot(x: np.ndarray, faulty_datapoints: int) -> bool:
    test_start_idx = x.shape[0] - (faulty_datapoints + 1)
    train, test = x[:test_start_idx], x[test_start_idx:]
    alarms = spot(train, test, proba=1e-4, n_points=10)[1]
    return alarms.size > 0


def find_records_detected_anomalies_of_sli(
    records: list[DatasetRecord],
    faulty_datapoints: int,
    sli_index: int = 0,
) -> list[DatasetRecord]:
    def _detect_anomalous_sli(record: DatasetRecord, faulty_datapoints: int, sli_index: int) -> bool:
        sli_name = record.pk.get_root_metrics()[sli_index]
        x = record.data_df[sli_name].to_numpy()
        return detect_anomalies_with_spot(x, faulty_datapoints)

    anomalous_record_idx: list[bool] = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_detect_anomalous_sli)(record, faulty_datapoints, sli_index) for record in records
    )
    assert anomalous_record_idx is not None
    return [r for r, is_anomalous in zip(records, anomalous_record_idx) if is_anomalous]


def find_records_detected_anomalies_of_cause_metrics(
    records: list[DatasetRecord],
    faulty_datapoints: int,
) -> list[DatasetRecord]:

    def _detect_anomalous_cause_metrics(record: DatasetRecord, faulty_datapoints: int) -> bool:
        ok, cause_metrics = check_cause_metrics(record.pk, record.data_df.columns.tolist(), record.chaos_type(), record.chaos_comp(), optional_cause=True)
        anomalies: list[bool] = []
        for metric in cause_metrics.tolist():
            x = record.data_df.loc[:, metric].to_numpy()
            anomalies.append(detect_anomalies_with_spot(x, faulty_datapoints))
        return any(anomalies)

    anomalous_record_idx = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_detect_anomalous_cause_metrics)(record, faulty_datapoints) for record in records
    )
    assert anomalous_record_idx is not None
    return [r for r, is_anomalous in zip(records, anomalous_record_idx) if is_anomalous]


def validate_anomalie_range(
    metrics: pd.DataFrame, labbeling: dict[str, dict[str, Any]], fi_time: int
) -> dict[int, Any]:
    """Evaluate the range of anomalies in KPI metrics"""
    result: dict[int, Any] = {}
    for n_sigma in labbeling["n_sigma_rule"]["n_sigmas"]:
        anomalies_range = metrics.apply(
            lambda X: detect_with_n_sigma_rule(X, test_start_time=fi_time, sigma_threshold=n_sigma).size > 0
        )
        result[n_sigma] = anomalies_range.to_dict()
    return result


def validate_route(
    record: DatasetRecord,
    route_no: int,
    gt_route,
    gt_route_matcher,
    labbeling: dict[str, dict[str, Any]],
    fault_inject_time_index: int,
) -> list[dict[str, Any]]:
    validation_results: list[dict[str, Any]] = []
    gt_route_metrics = record.data_df.loc[:, record.data_df.columns.intersection(set(gt_route))]
    res = validate_anomalie_range(
        gt_route_metrics,
        labbeling,
        fi_time=fault_inject_time_index,
    )
    for n, val in res.items():
        ok_kpis: list[str] = [kpi for kpi, ok in val.items() if not math.isnan(ok) and ok]
        total_ok, _ = check_route(ok_kpis, gt_route_matcher)
        validation_results.append(
            dict(
                {
                    "chaos_type": record.chaos_type(),
                    "chaos_comp": record.chaos_comp(),
                    "metrics_file": record.basename_of_metrics_file(),
                    "route_no": route_no,
                    "n_sigma": n,
                    "ok": total_ok,
                },
                **val,
            )
        )
    return validation_results


def validate_data_record(
    record: DatasetRecord,
    labbeling: dict[str, dict[str, Any]],
    fault_inject_time_index: int,
) -> pd.DataFrame | None:
    gt_metrics_routes = select_ground_truth_metrics_in_routes(
        record.pk,
        list(record.data_df.columns),
        record.chaos_type(),
        record.chaos_comp(),
        {
            "cause_service": True,
            "cause_middleware": False,
            "neighbors_in_cause_service": False,
            "propagated_route": False,
        },
    )
    validation_results: list[dict[str, Any]] = []
    for i, (gt_route, gt_route_matcher) in enumerate(gt_metrics_routes):
        res = validate_route(record, i, gt_route, gt_route_matcher, labbeling, fault_inject_time_index)
        validation_results.extend(res)
    if len(validation_results) == 0:
        return None
    return pd.DataFrame(validation_results).set_index(
        ["chaos_type", "chaos_comp", "metrics_file", "route_no", "n_sigma"]
    )


def check_valid_dataset(
    record: DatasetRecord,
    labbeling: dict[str, Any],
    fault_inject_time_index: int,
) -> bool:
    df: pd.DataFrame | None = validate_data_record(record, labbeling, fault_inject_time_index)
    if df is None:
        return False
    return df.loc[(record.chaos_type(), record.chaos_comp(), record.basename_of_metrics_file()), :]["ok"].any()
