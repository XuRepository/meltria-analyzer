#!/usr/bin/env python3

import logging
import os
from functools import partial
from typing import Any

import hydra
import joblib
import neptune.new as neptune
import pandas as pd
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate

import meltria.loader
from meltria.loader import DatasetRecord
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule
from tsdr.tsdr import filter_out_no_change_metrics

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger('eval_dataset')
logger.setLevel(logging.INFO)


def detect_anomalies_in_record(
    record: DatasetRecord, labbeling: dict[str, dict[str, Any]], fi_time: int,
) -> pd.DataFrame:
    logger.info(f">> Processing {record.chaos_case_full()} ...")

    filtered_df: pd.DataFrame = filter_out_no_change_metrics(record.data_df)

    """ Detect anomalies in a dataset record """
    def detect_anomaly(X: pd.Series, n_sigma: int) -> bool:
        return detect_with_n_sigma_rule(X, test_start_time=fi_time, sigma_threshold=n_sigma).size > 0

    items: list[dict[str, str | int | float]] = []
    for n_sigma in labbeling['n_sigma_rule']['n_sigmas']:
        f = partial(detect_anomaly, n_sigma=n_sigma)
        vc: pd.Series = record.data_df.apply(f).value_counts()
        items.append({
            'target_app': record.target_app(),
            'chaos_type': record.chaos_type(),
            'chaos_comp': record.chaos_comp(),
            'chaos_case_num': record.chaos_case_num(),
            'n_sigma': n_sigma,
            'num_anomalies': f"{vc[True]}/{filtered_df.shape[1]}/{vc[True] + vc[False]}",
            'anomalies_rate': round(vc[True] / filtered_df.shape[1], 2),
        })
    df = pd.DataFrame(
        items,
        columns=['target_app', 'chaos_type', 'chaos_comp', 'chaos_case_num', 'n_sigma', 'num_anomalies', 'anomalies_rate'],
    ).set_index(['target_app', 'chaos_type', 'chaos_comp', 'chaos_case_num', 'n_sigma'])
    return df


def eval_dataset(run: neptune.Run, cfg: DictConfig) -> None:
    """ Evaluate a dataset """
    dataset_generator = meltria.loader.load_dataset_as_generator(
        cfg.metrics_files,
        OmegaConf.to_container(cfg.target_metric_types, resolve=True),
        cfg.time.num_datapoints,
    )
    logger.info(">> Loading dataset")

    labbeling = OmegaConf.to_container(cfg.labbeling, resolve=True)
    fi_time = cfg.time.fault_inject_time_index

    stat_df_list: list[pd.DataFrame] = []
    for records in dataset_generator:
        dfs: list[pd.DataFrame] | None = joblib.Parallel(n_jobs=-1, backend='multiprocessing')(
            joblib.delayed(detect_anomalies_in_record)(record, labbeling, fi_time) for record in records
        )
        if dfs is None:
            continue
        stat_df_list.extend(dfs)

    stat_df = pd.concat(stat_df_list, copy=False)
    # reset_index: see https://stackoverflow.com/questions/70013696/print-multi-index-dataframe-with-tabulate
    print(tabulate(
        stat_df.reset_index(), headers='keys', tablefmt='github',
        numalign='right', stralign='left', showindex='always'))


@hydra.main(version_base="1.2", config_path='../conf/dataset', config_name='config')
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
