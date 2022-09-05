import math
from concurrent import futures
from concurrent.futures import Future
from multiprocessing import cpu_count
from typing import Any

import pandas as pd

from eval.groundtruth import check_route, select_ground_truth_metrics_in_routes
from meltria.loader import DatasetRecord
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule


def validate_anomalie_range(metrics: pd.DataFrame, labbeling: dict[str, Any], fi_time: int) -> dict[int, Any]:
    """ Evaluate the range of anomalies in KPI metrics """
    result: dict[int, Any] = {}
    for n_sigma in labbeling['n_sigma_rule']['n_sigmas']:
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
    labbeling: dict[str, Any],
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
        validation_results.append(dict({
            'chaos_type': record.chaos_type(),
            'chaos_comp': record.chaos_comp(),
            'metrics_file': record.basename_of_metrics_file(),
            'route_no': route_no,
            'n_sigma': n,
            'ok': total_ok,
        }, **val))
    return validation_results


def validate_data_record(
    record: DatasetRecord,
    labbeling: dict[str, Any],
    fault_inject_time_index: int,
    n_workers: int = 0,
) -> pd.DataFrame | None:
    gt_metrics_routes = select_ground_truth_metrics_in_routes(
        record.pk, list(record.data_df.columns), record.chaos_type(), record.chaos_comp(),
    )
    validation_results: list[dict[str, Any]] = []
    if n_workers == 0:
        n_workers = cpu_count()
    with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_gt_route_no: dict[Future, int] = {}
        for i, (gt_route, gt_route_matcher) in enumerate(gt_metrics_routes):
            f: Future = executor.submit(
                validate_route, record, i, gt_route, gt_route_matcher, labbeling, fault_inject_time_index)
            future_to_gt_route_no[f] = i
        for future in futures.as_completed(future_to_gt_route_no):
            if (res := future.result()) is None:
                continue
            validation_results.extend(res)
    if not validation_results:
        return None
    return pd.DataFrame(validation_results).set_index(['chaos_type', 'chaos_comp', 'metrics_file', 'route_no', 'n_sigma'])


def check_valid_dataset(
    record: DatasetRecord,
    labbeling: dict[str, Any],
    fault_inject_time_index: int,
) -> bool:
    df: pd.DataFrame | None = validate_data_record(record, labbeling, fault_inject_time_index)
    if df is None:
        return False
    return df.loc[(record.chaos_type(), record.chaos_comp(), record.basename_of_metrics_file()), :]['ok'].any()
