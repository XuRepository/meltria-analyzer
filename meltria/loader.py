import json
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterator
from concurrent import futures
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any

import numpy as np
import pandas as pd

from eval import groundtruth
from meltria.priorknowledge.priorknowledge import PriorKnowledge, new_knowledge

METRIC_TYPE_SERVICES: str = 'services'
METRIC_TYPE_CONTAINERS: str = 'containers'
METRIC_TYPE_NODES: str = 'nodes'
METRIC_TYPE_MIDDLEWARES: str = 'middlewares'

METRIC_TYPE_MAP: list[tuple[str, str]] = [
    ('c-', METRIC_TYPE_CONTAINERS), ('s-', METRIC_TYPE_SERVICES), ('m-', METRIC_TYPE_MIDDLEWARES), ('n-', METRIC_TYPE_NODES)
]


@dataclass
class DatasetRecord:
    """A record of dataset"""
    data_df: pd.DataFrame
    pk: PriorKnowledge
    meta: dict[str, Any]
    metrics_file: str  # path of metrics file eg. '2021-12-09-argowf-chaos-hg68n-carts-db_pod-cpu-hog_4.json'

    def target_app(self) -> str:
        """target-application eg. 'train-ticket'"""
        return self.meta['target_app']

    def chaos_comp(self) -> str:
        """chaos-injected component eg. 'carts-db'"""
        return self.meta['chaos_injected_component']

    def chaos_type(self) -> str:
        """injected chaos type eg. 'pod-cpu-hog'"""
        return self.meta['injected_chaos_type']

    def grafana_dashboard_url(self) -> str:
        return self.meta['grafana_dashboard_url']

    def chaos_case(self) -> str:
        return f"{self.chaos_comp()}/{self.chaos_type()}"

    def chaos_case_full(self) -> str:
        return f"{self.chaos_case()}/{self.chaos_case_num()}"

    def chaos_case_file(self) -> str:
        return f"{self.basename_of_metrics_file()} of {self.chaos_case()}"

    def chaos_case_num(self) -> str:
        # eg. '2021-12-09-argowf-chaos-hg68n-carts-db_pod-cpu-hog_4.json'
        return self.basename_of_metrics_file().rsplit('_', maxsplit=1)[1].removesuffix('.json')

    def metrics_names(self) -> list[str]:
        return list(self.data_df.columns)

    def basename_of_metrics_file(self) -> str:
        return os.path.basename(self.metrics_file)

    def ground_truth_metrics_frame(self) -> pd.DataFrame | None:
        _, ground_truth_metrics = groundtruth.check_tsdr_ground_truth_by_route(
            pk=self.pk,
            metrics=self.metrics_names(),  # pre-reduced data frame
            chaos_type=self.chaos_type(),
            chaos_comp=self.chaos_comp(),
        )
        if len(ground_truth_metrics) < 1:
            return None
        ground_truth_metrics.sort()
        return self.data_df[ground_truth_metrics]


def load_dataset_as_generator(
    metrics_files: list[str], target_metric_types: dict[str, bool], num_datapoints: int,
    n_jobs: int = 0,
) -> Iterator[list[DatasetRecord]]:
    """ Load n_jobs files at a time as generator """
    if n_jobs == 0:
        n_jobs = cpu_count()
    if len(metrics_files) < n_jobs:
        yield load_dataset(metrics_files, target_metric_types, num_datapoints)
    else:
        parts_of_files: list[np.ndarray] = np.array_split(metrics_files, int(len(metrics_files)/n_jobs))
        for part_of_files in parts_of_files:
            yield load_dataset(part_of_files.tolist(), target_metric_types, num_datapoints)


def load_dataset(
    metrics_files: list[str], target_metric_types: dict[str, bool], num_datapoints: int,
    logger: logging.Logger = logging.getLogger(),
) -> list[DatasetRecord]:
    """ Load metrics dataset """
    records: list[DatasetRecord] = []
    with futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        future_to_metrics_file = {}
        for metrics_file in metrics_files:
            f = executor.submit(read_metrics_file, metrics_file, target_metric_types, num_datapoints)
            future_to_metrics_file[f] = os.path.basename(metrics_file)
        for future in futures.as_completed(future_to_metrics_file):
            metrics_file = future_to_metrics_file[future]
            try:
                record: DatasetRecord = future.result()
            except ValueError as e:
                logger.warning(f">> Skip {metrics_file} because of {e}")
                continue
            records.append(record)
    if len(records) < 1:
        raise ValueError("No metrics data loaded")
    return records


def read_metrics_file(
    data_file: str,
    target_metric_types: dict[str, bool],
    num_datapoints: int,
    interporate: bool = True,
) -> DatasetRecord:
    """ Read metrics data file """
    with open(data_file) as f:
        raw_data: dict[str, Any] = json.load(f)
    pk: PriorKnowledge = new_knowledge(raw_data['meta']['target_app'], raw_data['mappings'])
    metrics_name_to_values: dict[str, np.ndarray] = {}
    for metric_type, enable in target_metric_types.items():
        if not enable:
            continue
        for metrics in raw_data[metric_type].values():
            for metric in metrics:
                # remove prefix of label name that Prometheus gives
                metric_name = metric["metric_name"].removeprefix("container_").removeprefix("node_")
                target_name = metric["{}_name".format(metric_type[:-1]) if metric_type != METRIC_TYPE_MIDDLEWARES else "container_name"]
                if metric_type == METRIC_TYPE_SERVICES:
                    if (service := pk.get_service_by_container(target_name)) is not None:
                        target_name = service
                elif metric_type == METRIC_TYPE_NODES:
                    # FIXME: workaround for node name include ';' as suffix due to the issue of prometheus config
                    target_name = target_name.removesuffix(';')
                if target_name in pk.get_skip_containers():
                    continue
                metric_name = "{}-{}_{}".format(metric_type[0], target_name, metric_name)
                metrics_name_to_values[metric_name] = np.array(
                    metric["values"], dtype=np.float64,
                )[:, 1][-num_datapoints:]
    data_df = pd.DataFrame(metrics_name_to_values).round(4)
    if interporate:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data_df.interpolate(method="akima", limit_direction="both", inplace=True)
        except:  # To cacth `dfitpack.error: (m>k) failed for hidden m: fpcurf0:m=3`
            raise ValueError("calculating spline error") from None
    return DatasetRecord(data_df=data_df, pk=pk, metrics_file=data_file, meta=raw_data['meta'])


def count_metrics(df: pd.DataFrame) -> pd.DataFrame:
    counter: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: 0))
    for col in df.columns:
        for prefix, metric_type in METRIC_TYPE_MAP:
            if col.startswith(prefix):
                comp_name = col.split('_')[0].removeprefix(prefix)
                counter[metric_type][comp_name] += 1
    clist = [{'metric_type': t, 'comp_name': n, 'count': cnt} for t, v in counter.items() for n, cnt in v.items()]
    return pd.DataFrame(clist).set_index(['metric_type', 'comp_name'])
