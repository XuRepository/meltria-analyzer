import json
import logging
import os
import warnings
from collections import defaultdict
from concurrent import futures
from multiprocessing import cpu_count
from typing import Any

import numpy as np
import pandas as pd

from eval import groundtruth
from meltria.priorknowledge.priorknowledge import PriorKnowledge

PLOTS_NUM = 120
TARGET_DATA = {
    "containers": "all",
    "services": "all",
    "nodes": "all",
    "middlewares": "all",
}


class DatasetRecord:
    """A record of dataset"""
    chaos_comp: str     # chaos-injected component
    chaos_type: str     # injected chaos type
    metrics_file: str   # path of metrics file
    data_df: pd.DataFrame

    def __init__(self, target_app: str, chaos_type: str, chaos_comp: str, metrics_file: str, data_df: pd.DataFrame):
        self.target_app = target_app
        self.chaos_comp = chaos_comp
        self.chaos_type = chaos_type
        self.metrics_file = metrics_file
        self.data_df = data_df

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


def load_dataset(
    metrics_files: list[str], exclude_middleware_metrics: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """ Load metrics dataset
    """
    df_list: list[pd.DataFrame] = []
    mappings_by_metrics_file: dict[str, Any] = {}
    with futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        future_to_metrics_file = {}
        for metrics_file in metrics_files:
            f = executor.submit(read_metrics_file, metrics_file, exclude_middleware_metrics)
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
    exclude_middleware_metrics: bool = False,
    logger: logging.Logger = logging.getLogger(),
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    try:
        data_df, mappings, metrics_meta = read_metrics_json(
            metrics_file,
            exclude_middlewares=exclude_middleware_metrics,
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
    interporate: bool = True,
    exclude_middlewares: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """ Read metrics data file """
    # FIXME: loading json twice should be avoided
    with open(data_file) as f:
        raw_json = json.load(f)
    raw_data = pd.read_json(data_file)
    data_df = pd.DataFrame()
    metrics_name_to_values: dict[str, np.ndarray] = {}
    for target in TARGET_DATA:
        for t in raw_data[target].dropna():
            for metric in t:
                if metric["metric_name"] not in TARGET_DATA[target] and TARGET_DATA[target] != "all":
                    continue
                if target == 'middlewares' and exclude_middlewares:
                    continue
                metric_name = metric["metric_name"].replace(
                    "container_", "").replace("node_", "")
                target_name = metric["{}_name".format(target[:-1]) if target != "middlewares" else "container_name"]
                if target_name in ["queue-master", "rabbitmq", "session-db"]:
                    continue
                metric_name = "{}-{}_{}".format(target[0], target_name, metric_name)
                metrics_name_to_values[metric_name] = np.array(
                    metric["values"], dtype=np.float64,
                )[:, 1][-PLOTS_NUM:]
    data_df = pd.DataFrame(metrics_name_to_values).round(4)
    if interporate:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data_df = data_df.interpolate(method="spline", order=3, limit_direction="both")
        except:  # To cacth `dfitpack.error: (m>k) failed for hidden m: fpcurf0:m=3`
            raise ValueError("calculating spline error") from None
    return data_df, raw_json['mappings'], raw_json['meta']


def count_metrics(df: pd.DataFrame) -> pd.DataFrame:
    map_type: list[tuple[str, str]] = [
        ('c-', 'containers'), ('s-', 'services'), ('m-', 'middlewares'), ('n-', 'nodes')]
    counter: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: 0))
    for col in df.columns:
        for prefix, comp_type in map_type:
            if col.startswith(prefix):
                comp_name = col.split('_')[0].removeprefix(prefix)
                counter[comp_type][comp_name] += 1
    clist = [{'comp_type': t, 'comp_name': n, 'count': cnt} for t, v in counter.items() for n, cnt in v.items()]
    return pd.DataFrame(clist).set_index(['comp_type', 'comp_name'])
