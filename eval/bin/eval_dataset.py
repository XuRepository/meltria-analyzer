#!/usr/bin/env python3

import logging
import math
import os
from typing import Any

import hydra
import neptune.new as neptune
import pandas as pd
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate

import meltria.loader
from eval.groundtruth import check_route, select_ground_truth_metrics_in_routes
from meltria.loader import DatasetRecord
from meltria.priorknowledge.priorknowledge import PriorKnowledge, new_knowledge
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger('eval_dataset')
logger.setLevel(logging.INFO)


def validate_anomalie_range(metrics: pd.DataFrame, labbeling: dict[str, Any], fi_time: int) -> dict[int, Any]:
    """ Evaluate the range of anomalies in KPI metrics """
    result: dict[int, Any] = {}
    for n_sigma in labbeling['n_sigma_rule']['n_sigmas']:
        anomalies_range = metrics.apply(
            lambda X: detect_with_n_sigma_rule(X, test_start_time=fi_time, sigma_threshold=n_sigma).size > 0
        )
        result[n_sigma] = anomalies_range.to_dict()
    return result


def eval_dataset(run: neptune.Run, cfg: DictConfig) -> None:
    """ Evaluate a dataset """
    dataset: pd.DataFrame = meltria.loader.load_dataset(
        cfg.metrics_files,
        OmegaConf.to_container(cfg.target_metric_types, resolve=True),
        cfg.time.num_datapoints,
    )[0]
    logger.info("Dataset loading complete")

    kpi_anomalies: list[dict[str, Any]] = []
    for (target_app, chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1, 2]):
        prior_knowledge: PriorKnowledge = new_knowledge(target_app)
        for (metrics_file, grafana_dashboard_url), data_df in sub_df.groupby(level=[3, 4]):
            record = DatasetRecord(target_app, chaos_type, chaos_comp, metrics_file, data_df)

            gt_metrics_routes = select_ground_truth_metrics_in_routes(
                prior_knowledge, list(data_df.columns), chaos_type, chaos_comp,
            )
            for i, (gt_route, gt_route_matcher) in enumerate(gt_metrics_routes):
                gt_route_metrics = data_df.loc[:, data_df.columns.intersection(set(gt_route))]
                res = validate_anomalie_range(
                    gt_route_metrics,
                    OmegaConf.to_container(cfg.labbeling, resolve=True),
                    fi_time=cfg.time.fault_inject_time_index,
                )
                for n, val in res.items():
                    ok_kpis: list[str] = [kpi for kpi, ok in val.items() if not math.isnan(ok) and ok]
                    total_ok, _ = check_route(ok_kpis, gt_route_matcher)
                    kpi_anomalies.append(dict({
                        'chaos_type': record.chaos_type,
                        'chaos_comp': record.chaos_comp,
                        'metrics_file': record.metrics_file,
                        'route_no': i,
                        'n_sigma': n,
                        'ok': total_ok,
                    }, **val))

    kpi_df = pd.DataFrame(kpi_anomalies).set_index(['chaos_type', 'chaos_comp', 'metrics_file', 'route_no', 'n_sigma'])
    # reset_index: see https://stackoverflow.com/questions/70013696/print-multi-index-dataframe-with-tabulate
    print(tabulate(
        kpi_df.reset_index(), headers='keys', tablefmt='github',
        numalign='right', stralign='left', showindex='always'))


@hydra.main(config_path='../conf/dataset', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(
        project=os.environ['TSDR_NEPTUNE_PROJECT'],
        api_token=os.environ['TSDR_NEPTUNE_API_TOKEN'],
        mode=cfg.neptune.mode,
    )
    npt_handler = NeptuneHandler(run=run)
    logger.addHandler(npt_handler)

    run['dataset/id'] = cfg.dataset_id
    run['dataset/num_metrics_files'] = len(cfg.metrics_files)
    params = {
        'target_metric_types': OmegaConf.to_container(cfg.target_metric_types, resolve=True),
    }

    # Hydra parameters are passed to the Neptune.ai run object
    params.update(OmegaConf.to_container(cfg, resolve=True))

    run['parameters'] = params
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_dataset(run, cfg)

    run.stop()


if __name__ == '__main__':
    main()
