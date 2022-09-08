#!/usr/bin/env python3

import logging
import os
from collections.abc import Callable
from typing import Any

import hydra
import joblib
import neptune.new as neptune
import pandas as pd
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate

import meltria.loader
from eval.labeling import detect_anomalies_in_record
from eval.validation import validate_data_record

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger("eval_dataset")
logger.setLevel(logging.INFO)


def eval_dataset(run: neptune.Run, cfg: DictConfig) -> None:
    """Evaluate a dataset"""
    dataset_generator = meltria.loader.load_dataset_as_generator(
        cfg.metrics_files,
        OmegaConf.to_container(cfg.target_metric_types, resolve=True),
        cfg.time.num_datapoints,
    )
    logger.info(">> Loading dataset")

    labbeling: dict[str, dict[str, Any]] = OmegaConf.to_container(cfg.labbeling, resolve=True)
    fi_time: int = cfg.time.fault_inject_time_index

    eval_func: Callable
    doing: str
    match cfg.eval_task:
        case "validation":
            eval_func, doing = validate_data_record, "Validating"
        case "labeling":
            eval_func, doing = detect_anomalies_in_record, "Labeling"
        case _:
            raise ValueError(f"Unknown eval_task: {cfg.eval_task}")

    stat_dfs: list[pd.DataFrame] = []
    for records in dataset_generator:
        for record in records:
            logger.info(f">> {doing} data record of {record.chaos_case_full()} ...")

        dfs: list[pd.DataFrame] | None = joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
            joblib.delayed(eval_func)(record, labbeling, fi_time) for record in records
        )
        if dfs is None:
            continue
        stat_dfs.extend(dfs)

    stat_df = pd.concat(stat_dfs)
    # reset_index: see https://stackoverflow.com/questions/70013696/print-multi-index-dataframe-with-tabulate
    print(
        tabulate(
            stat_df.reset_index(),
            headers="keys",
            tablefmt="github",
            numalign="right",
            stralign="left",
            showindex="always",
        )
    )


@hydra.main(version_base="1.2", config_path="../conf/dataset", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(
        project=os.environ["DATASET_NEPTUNE_PROJECT"],
        api_token=os.environ["DATASET_NEPTUNE_API_TOKEN"],
        mode=cfg.neptune.mode,
    )
    npt_handler = NeptuneHandler(run=run)
    logger.addHandler(npt_handler)

    run["dataset/id"] = cfg.dataset_id
    run["dataset/num_metrics_files"] = len(cfg.metrics_files)
    params = {
        "target_metric_types": OmegaConf.to_container(cfg.target_metric_types, resolve=True),
    }

    # Hydra parameters are passed to the Neptune.ai run object
    params.update(OmegaConf.to_container(cfg, resolve=True))

    run["parameters"] = params
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_dataset(run, cfg)

    run.stop()


if __name__ == "__main__":
    main()
