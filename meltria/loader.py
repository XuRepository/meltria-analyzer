import json
import warnings
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from eval import validation
from eval.util.logger import logger
from meltria.dataset import DatasetRecord
from meltria.metric_types import (METRIC_TYPE_MAP, METRIC_TYPE_MIDDLEWARES,
                                  METRIC_TYPE_NODES, METRIC_TYPE_SERVICES)
from meltria.priorknowledge.priorknowledge import PriorKnowledge, new_knowledge


def find_metrics_files(root_path: str, dataset_id: str) -> list[str]:
    """Find metrics files"""
    path = Path(root_path) / f"argowf-chaos-{dataset_id}"
    metrics_files: list[str] = []
    for file in path.glob("**/*.json"):
        metrics_files.append(file.absolute().as_posix())
    return metrics_files


def load_dataset_as_generator(
    metrics_files: list[str],
    target_metric_types: dict[str, bool],
    num_datapoints: int,
    interpolate: bool = True,
    n_jobs: int = -1,
) -> Iterator[list[DatasetRecord]]:
    """Load n_jobs files at a time as generator"""
    if len(metrics_files) < n_jobs:
        yield load_dataset(
            metrics_files, target_metric_types, num_datapoints, interpolate, n_jobs
        )
    else:
        parts_of_files: list[np.ndarray] = np.array_split(
            metrics_files, int(len(metrics_files) / n_jobs)
        )
        for part_of_files in parts_of_files:
            yield load_dataset(
                part_of_files.tolist(), target_metric_types, num_datapoints, interpolate
            )


def load_dataset(
    metrics_files: list[str],
    target_metric_types: dict[str, bool],
    num_datapoints: int,
    interpolate: bool = True,
    validated: bool = False,
    max_chaos_case_num: int = -1,
    num_faulty_datapoints: int = 0,
    target_chaos_types: set[str] = set(),  # empty means all
    n_jobs: int = -1,
) -> list[DatasetRecord]:
    """Load metrics dataset"""
    records: list[DatasetRecord] | None = joblib.Parallel(
        n_jobs=n_jobs, backend="multiprocessing"
    )(
        joblib.delayed(read_metrics_file)(
            path, target_metric_types, num_datapoints, interpolate
        )
        for path in metrics_files
    )
    if records is None or len(records) < 1:
        raise ValueError("No metrics data loaded")
    records = [r for r in records if r is not None]
    if len(target_chaos_types):
        records = [r for r in records if r.chaos_type() in target_chaos_types]

    if not validated:
        return select_records_within_litmit_num(records, max_chaos_case_num=max_chaos_case_num)

    well_injected_records = validation.find_records_detected_anomalies_of_sli(
        records,
        faulty_datapoints=num_faulty_datapoints,
    )
    well_injected_records = validation.find_records_detected_anomalies_of_cause_metrics(
        well_injected_records,
        faulty_datapoints=num_faulty_datapoints,
        optional_cause=False,
    )
    return balance_records_by_removing(well_injected_records, lower_limit_num=max_chaos_case_num)


def transform_records_to_dict(records: list[DatasetRecord]) -> dict[tuple[str, str], list[DatasetRecord]]:
    """Transform records to dict"""
    chaos_cases: dict[tuple[str, str], list[DatasetRecord]] = defaultdict(list)
    for record in records:
        chaos_cases[(record.chaos_type(), record.chaos_comp())].append(record)
    return chaos_cases


def select_records_within_litmit_num(records: list[DatasetRecord], max_chaos_case_num: int) -> list[DatasetRecord]:
    if max_chaos_case_num < 0:
        return records
    selected_records_ = []
    chaos_cases = transform_records_to_dict(records)
    for _, records_ in chaos_cases.items():
        if len(records_) > max_chaos_case_num:
            selected_records_.extend(records_[:max_chaos_case_num])
        else:
            selected_records_.extend(records_)
    return selected_records_


def balance_records_by_removing(records: list[DatasetRecord], lower_limit_num: int) -> list[DatasetRecord]:
    if lower_limit_num < -1:
        return records

    chaos_cases: dict[tuple[str, str], list[DatasetRecord]] = defaultdict(list)
    chaos_types: set[str] = set()
    chaos_comps: set[str] = set()
    for record in records:
        chaos_cases[(record.chaos_type(), record.chaos_comp())].append(record)
        chaos_types.add(record.chaos_type())
        chaos_comps.add(record.chaos_comp())

    selected_records_ = []
    for chaos_comp in chaos_comps:
        chaos_type_to_num = defaultdict(int)
        for chaos_type in chaos_types:
            chaos_type_to_num[chaos_type] = len(chaos_cases[(chaos_type, chaos_comp)])
        if min(chaos_type_to_num.values()) < lower_limit_num:
            continue
        for chaos_type in chaos_types:
            selected_records_.extend(chaos_cases[(chaos_type, chaos_comp)][:lower_limit_num])
    return selected_records_


RANGE_VECTOR_DURATION = 60
PER_MINUTE_NUM: int = int(RANGE_VECTOR_DURATION / 15) + 1


def rate_of_metrics(ts: np.ndarray) -> np.ndarray:
    slides = np.lib.stride_tricks.sliding_window_view(ts, PER_MINUTE_NUM)
    rate = (
        np.max(slides, axis=1).reshape(-1) - np.min(slides, axis=1).reshape(-1)
    ) / RANGE_VECTOR_DURATION
    first_val = rate[0]
    for _ in range(PER_MINUTE_NUM - 1):
        rate = np.insert(rate, 0, first_val)  # backfill
    return rate  # type: ignore


def is_monotonic_increasing(x: np.ndarray) -> bool:
    for i in range(1, len(x)):
        if x[i - 1] > x[i]:
            if x[i] == 0.0:  # detect reset
                continue
            else:
                return False
    return True


def is_counter(x: np.ndarray) -> bool:
    return bool(
        not np.all(x == x[0])  # check all values are the same
        and np.any(x > 0)  # check all values are positive
        and is_monotonic_increasing(x)  # check monotonic increasing
        and np.any(
            x == np.round(x)
        )  # check including float because a counter metric should be integer.
    )


def check_counter_and_rate(ts: np.ndarray) -> np.ndarray:
    if not is_counter(ts):
        return ts
    rated_ts = rate_of_metrics(ts)
    if np.any(rated_ts < 0.0):  # return the unrated series if the rated series has a negative value.
        return ts
    return rated_ts


EXCLUDE_PROMETHEUS_DEFAULT_METRICS: Final[set[str]] = {
    "process_cpu_seconds_total",
    "process_cpu_seconds_total",
    "process_virtual_memory_bytes",
    "process_resident_memory_bytes",
    "process_start_time_seconds",
    "process_open_fds",
    "process_max_fds",
    "promhttp_metric_handler_requests_in_flight",
    "promhttp_metric_handler_requests_total",
    "go_goroutines",
    "go_threads",
    "go_gc_duration_seconds",
    "go_gc_cycles_automatic_gc_cycles_total",
    "go_gc_cycles_forced_gc_cycles_total",
    "go_gc_cycles_total_gc_cycles_total",
    "go_gc_heap_allocs_by_size_bytes_total_bucket",
    "go_gc_heap_allocs_by_size_bytes_total_sum",
    "go_gc_heap_allocs_by_size_bytes_total_count",
    "go_gc_heap_allocs_bytes_total",
    "go_gc_heap_allocs_objects_total",
    "go_gc_heap_frees_by_size_bytes_total_bucket",
    "go_gc_heap_frees_by_size_bytes_total_sum",
    "go_gc_heap_frees_by_size_bytes_total_count",
    "go_gc_heap_frees_bytes_total",
    "go_gc_heap_frees_objects_total",
    "go_gc_heap_goal_bytes",
    "go_gc_heap_objects_objects",
    "go_gc_heap_tiny_allocs_objects_total",
    "go_gc_pauses_seconds_total_bucket",
    "go_gc_pauses_seconds_total_sum",
    "go_gc_pauses_seconds_total_count",
    "go_info",
    "go_memory_classes_heap_free_bytes",
    "go_memory_classes_heap_objects_bytes",
    "go_memory_classes_heap_released_bytes",
    "go_memory_classes_heap_stacks_bytes",
    "go_memory_classes_heap_unused_bytes",
    "go_memory_classes_metadata_mcache_free_bytes",
    "go_memory_classes_metadata_mcache_inuse_bytes",
    "go_memory_classes_metadata_mspan_free_bytes",
    "go_memory_classes_metadata_mspan_inuse_bytes",
    "go_memory_classes_metadata_other_bytes",
    "go_memory_classes_os_stacks_bytes",
    "go_memory_classes_other_bytes",
    "go_memory_classes_profiling_buckets_bytes",
    "go_memory_classes_total_bytes",
    "go_memstats_alloc_bytes",
    "go_memstats_alloc_bytes_total",
    "go_memstats_sys_bytes",
    "go_memstats_lookups_total",
    "go_memstats_mallocs_total",
    "go_memstats_frees_total",
    "go_memstats_heap_alloc_bytes",
    "go_memstats_heap_sys_bytes",
    "go_memstats_heap_idle_bytes",
    "go_memstats_heap_idle_bytes",
    "go_memstats_heap_inuse_bytes",
    "go_memstats_heap_inuse_bytes",
    "go_memstats_heap_released_bytes",
    "go_memstats_heap_released_bytes_total",
    "go_memstats_heap_objects",
    "go_memstats_stack_inuse_bytes",
    "go_memstats_stack_sys_bytes",
    "go_memstats_stack_inuse_bytes",
    "go_memstats_stack_idle_bytes",
    "go_memstats_heap_idle_bytes",
    "go_memstats_mspan_inuse_bytes",
    "go_memstats_mspan_sys_bytes",
    "go_memstats_mcache_inuse_bytes",
    "go_memstats_mcache_sys_bytes",
    "go_memstats_buck_hash_sys_bytes",
    "go_memstats_gc_sys_bytes",
    "go_memstats_other_sys_bytes",
    "go_memstats_next_gc_bytes",
    "go_memstats_last_gc_time_seconds",
    "go_memstats_last_gc_cpu_fraction",
    "go_sched_goroutines_goroutines",
    "go_sched_latencies_seconds_bucket",
    "go_sched_latencies_seconds_sum",
    "go_sched_latencies_seconds_count",
    # for sock shop
    "microservices_demo_user_request_latency",
    "microservices_demo_user_request_latency_microseconds",
    "microservices_demo_user_request_count",
    # JVM
    "jvm_classes_loaded_total",
    "nonheap_used",
    "heap_used",
    "mem",
    "mem_free",
    "classes",
    "classes_loaded",
    "threads_totalStarted",
    "threads_daemon",
    "jvm_threads_daemon",
    "jvm_threads_current",
    "jvm_memory_pool_bytes_committed",
    "jvm_memory_bytes_used",
    "org_mongodb_driver_ConnectionPool_CheckedOutCount",
    "org_mongodb_driver_ConnectionPool_Size",
    "org_springframework_cloud_context_restart_restartEndpoint_Running",
    "uptime",
    "instance_uptime",
    "gc_g1_young_generation_count",
    "gauge_response_health",
    "threads_peak",
    "systemload_average",
    "jvm_threads_peak",
    "jvm_threads_started_total",
    "jmx_scrape_duration_seconds",
    "jmx_scrape_error",
    "gauge_response_metrics",
    "gauge_response_orders",
    "counter_status_200_health",
    "counter_status_201_orders",
    "counter_status_406_orders",
}


def is_excluded_metrics(metric_base_name: str) -> bool:
    # bad side effects: golang application metrics are also excluded.
    return metric_base_name in EXCLUDE_PROMETHEUS_DEFAULT_METRICS


def update_count_of_meta(data_df: pd.DataFrame, meta: dict[str, Any]) -> None:
    """Update meta data"""
    meta["count"]["containers"] = data_df.loc[
        :, data_df.columns.str.startswith("c-")
    ].shape[1]
    meta["count"]["middlewares"] = data_df.loc[
        :, data_df.columns.str.startswith("m-")
    ].shape[1]
    meta["count"]["services"] = data_df.loc[
        :, data_df.columns.str.startswith("s-")
    ].shape[1]
    meta["count"]["nodes"] = data_df.loc[:, data_df.columns.str.startswith("n-")].shape[
        1
    ]
    assert data_df.shape[1] == (
        meta["count"]["containers"]
        + meta["count"]["middlewares"]
        + meta["count"]["nodes"]
        + meta["count"]["services"]
    )


def read_metrics_file(
    data_file: str,
    target_metric_types: dict[str, bool],
    num_datapoints: int,
    interporate: bool = True,
) -> DatasetRecord | None:
    """Read metrics data file"""
    with open(data_file) as f:
        raw_data: dict[str, Any] = json.load(f)
    pk: PriorKnowledge = new_knowledge(
        raw_data["meta"]["target_app"], target_metric_types, raw_data["mappings"]
    )
    metrics_name_to_values: dict[str, np.ndarray] = {}
    for metric_type, enable in target_metric_types.items():
        if not enable:
            continue
        for metrics in raw_data[metric_type].values():
            for metric in metrics:
                # Remove prefix of label name that Prometheus gives
                metric_name = metric["metric_name"].removeprefix("container_").removeprefix("node_")
                # Skip metrics of Prometheus exporter itself.
                if is_excluded_metrics(metric_name):
                    continue
                target_name = metric[
                    "{}_name".format(metric_type[:-1]) if metric_type != METRIC_TYPE_MIDDLEWARES else "container_name"
                ]
                if metric_type == METRIC_TYPE_SERVICES:
                    if service := pk.get_service_by_container_or_empty(target_name):
                        target_name = service
                elif metric_type == METRIC_TYPE_NODES:
                    # FIXME: workaround for node name include ';' as suffix due to the issue of prometheus config
                    target_name = target_name.removesuffix(";")
                if target_name in pk.get_skip_containers():
                    continue
                ts = np.array(metric["values"], dtype=np.float64)[:, 1][-num_datapoints:]
                if metric_type in [METRIC_TYPE_MIDDLEWARES, METRIC_TYPE_NODES]:
                    ts = check_counter_and_rate(ts)
                metric_name = "{}-{}_{}".format(metric_type[0], target_name, metric_name)
                metrics_name_to_values[metric_name] = ts

    try:
        data_df = pd.DataFrame(metrics_name_to_values).round(4)
    except ValueError as e:
        raise ValueError(f"Reading file {data_file}: {e}")
    if interporate:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # The interpolated time serie should not have negative values.
                # See https://stackoverflow.com/questions/40072420/interpolate-without-having-negative-values-in-python.
                data_df.interpolate(
                    method="pchip", limit_direction="both", inplace=True
                )
        except:  # To cacth `dfitpack.error: (m>k) failed for hidden m: fpcurf0:m=3`
            logger.warning(f"calculating spline error: {data_file}")
            return None
        # Set negative values to 0.0 because interpolating may cause negative values.
        data_df.where(data_df >= 0.0, 0.0, inplace=True)

    update_count_of_meta(
        data_df, raw_data["meta"]
    )  # update the number of metrics because some metrics are removed.
    return DatasetRecord(
        data_df=data_df, pk=pk, metrics_file=data_file, meta=raw_data["meta"]
    )


def count_metrics(df: pd.DataFrame) -> pd.DataFrame:
    counter: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: 0))
    for col in df.columns:
        for prefix, metric_type in METRIC_TYPE_MAP:
            if col.startswith(prefix):
                comp_name = col.split("_")[0].removeprefix(prefix)
                counter[metric_type][comp_name] += 1
    clist = [
        {"metric_type": t, "comp_name": n, "count": cnt}
        for t, v in counter.items()
        for n, cnt in v.items()
    ]
    return pd.DataFrame(clist).set_index(["metric_type", "comp_name"])
