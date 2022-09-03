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
    target_app: str     # target-application eg. 'train-ticket'
    chaos_comp: str     # chaos-injected component eg. 'carts-db'
    chaos_type: str     # injected chaos type eg. 'pod-cpu-hog'
    metrics_file: str   # path of metrics file eg. '2021-12-09-argowf-chaos-hg68n-carts-db_pod-cpu-hog_4.json'
    data_df: pd.DataFrame

    def chaos_case(self) -> str:
        return f"{self.chaos_comp}/{self.chaos_type}"

    def chaos_case_full(self) -> str:
        return f"{self.chaos_case()}/{self.chaos_case_num()}"

    def chaos_case_file(self) -> str:
        return f"{self.metrics_file} of {self.chaos_case()}"

    def chaos_case_num(self) -> str:
        # eg. '2021-12-09-argowf-chaos-hg68n-carts-db_pod-cpu-hog_4.json'
        return self.metrics_file.rsplit('_', maxsplit=1)[1].removesuffix('.json')

    def metrics_names(self) -> list[str]:
        return list(self.data_df.columns)

    def basename_of_metrics_file(self) -> str:
        return os.path.basename(self.metrics_file)

    def ground_truth_metrics_frame(self, pk: PriorKnowledge) -> pd.DataFrame | None:
        _, ground_truth_metrics = groundtruth.check_tsdr_ground_truth_by_route(
            pk=pk,
            metrics=self.metrics_names(),  # pre-reduced data frame
            chaos_type=self.chaos_type,
            chaos_comp=self.chaos_comp,
        )
        if len(ground_truth_metrics) < 1:
            return None
        ground_truth_metrics.sort()
        return self.data_df[ground_truth_metrics]


def load_dataset_as_generator(
    metrics_files: list[str], target_metric_types: dict[str, bool], num_datapoints: int,
    n_jobs: int = 0,
) -> Iterator[tuple[pd.DataFrame, dict[str, Any]]]:
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """ Load metrics dataset """
    df_list: list[pd.DataFrame] = []
    mappings_by_metrics_file: dict[str, Any] = {}
    with futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        future_to_metrics_file = {}
        for metrics_file in metrics_files:
            f = executor.submit(read_metrics_file, metrics_file, target_metric_types, num_datapoints)
            future_to_metrics_file[f] = os.path.basename(metrics_file)
        for future in futures.as_completed(future_to_metrics_file):
            data_df, mappings = future.result()
            if data_df is not None:
                df_list.append(data_df)
                metrics_file = future_to_metrics_file[future]
                mappings_by_metrics_file[metrics_file] = mappings
    if len(df_list) < 1:
        raise ValueError("No metrics data loaded")
    elif len(df_list) == 1:
        dataset: pd.DataFrame = df_list[0]
    else:
        dataset: pd.DataFrame = pd.concat(df_list)
    return dataset.set_index(['target_app', 'chaos_type', 'chaos_comp', 'metrics_file', 'grafana_dashboard_url']), \
        mappings_by_metrics_file


def read_metrics_file(
    metrics_file: str,
    target_metric_types: dict[str, bool],
    num_datapoints: int,
    logger: logging.Logger = logging.getLogger(),
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    try:
        data_df, mappings, metrics_meta = read_metrics_json(
            metrics_file, target_metric_types, num_datapoints,
        )
    except ValueError as e:
        logger.warning(f">> Skip {metrics_file} because of {e}")
        return None, None
    data_df['target_app'] = metrics_meta['target_app']
    data_df['chaos_type'] = metrics_meta['injected_chaos_type']
    data_df['chaos_comp'] = metrics_meta['chaos_injected_component']
    data_df['metrics_file'] = os.path.basename(metrics_file)
    data_df['grafana_dashboard_url'] = metrics_meta['grafana_dashboard_url']
    return data_df, mappings


def read_metrics_json(
    data_file: str,
    target_metric_types: dict[str, bool],
    num_datapoints: int,
    interporate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """ Read metrics data file """
    with open(data_file) as f:
        raw_data: dict[str, Any] = json.load(f)
    metrics_name_to_values: dict[str, np.ndarray] = {}
    # TODO: return pk object and reuse it.
    pk: PriorKnowledge = new_knowledge(raw_data['meta']['target_app'], raw_data['mappings'])
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
    return data_df, raw_data['mappings'], raw_data['meta']


def count_metrics(df: pd.DataFrame) -> pd.DataFrame:
    counter: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: 0))
    for col in df.columns:
        for prefix, metric_type in METRIC_TYPE_MAP:
            if col.startswith(prefix):
                comp_name = col.split('_')[0].removeprefix(prefix)
                counter[metric_type][comp_name] += 1
    clist = [{'metric_type': t, 'comp_name': n, 'count': cnt} for t, v in counter.items() for n, cnt in v.items()]
    return pd.DataFrame(clist).set_index(['metric_type', 'comp_name'])
