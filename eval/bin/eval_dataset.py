#!/usr/bin/env python3

import logging
import os
from typing import Any

import hydra
import neptune.new as neptune
import pandas as pd
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate

import meltria.loader
from eval.validation import validate_data_record
from meltria.loader import DatasetRecord
from meltria.priorknowledge.priorknowledge import PriorKnowledge, new_knowledge
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger('eval_dataset')
logger.setLevel(logging.INFO)


def eval_dataset(run: neptune.Run, cfg: DictConfig) -> None:
    """ Evaluate a dataset """
    dataset_generator = meltria.loader.load_dataset_as_generator(
        cfg.metrics_files,
        OmegaConf.to_container(cfg.target_metric_types, resolve=True),
        cfg.time.num_datapoints,
    )
    logger.info(">> Loading dataset")

    kpi_df_list: list[pd.DataFrame] = []
    for records in dataset_generator:
        record: DatasetRecord
        for record in records:
            kpi_df = validate_data_record(
                record,
                OmegaConf.to_container(cfg.labbeling, resolve=True),
                cfg.time.fault_inject_time_index,
            )
            if kpi_df is None:
                continue
            kpi_df_list.append(kpi_df)

    kpi_df = pd.concat(kpi_df_list)
    # reset_index: see https://stackoverflow.com/questions/70013696/print-multi-index-dataframe-with-tabulate
    print(tabulate(
        kpi_df.reset_index(), headers='keys', tablefmt='github',
        numalign='right', stralign='left', showindex='always'))


@hydra.main(version_base="1.2", config_path='../conf/dataset', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(
        project=os.environ['DATASET_NEPTUNE_PROJECT'],
        api_token=os.environ['DATASET_NEPTUNE_API_TOKEN'],
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
