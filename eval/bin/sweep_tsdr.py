#!/usr/bin/env python3

import argparse
import logging
import os
import pathlib
import runpy

from eval import tsdr, validation
from eval.util import logger as internal_logger
from meltria import loader

DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "dataset" / "data"

internal_logger.logger.setLevel(logging.ERROR)  # Suppress internal logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)

# Suppress Neptune log messages to console
npt_logger = logging.getLogger("root_experiment")
npt_logger.setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(prog="sweep_tsdr")
    parser.print_usage = parser.print_help
    parser.add_argument('-c', '--config', type=str, required=True, help='config file')
    parser.add_argument('--num-datapoints', type=int, required=True, help='num datapoins')
    parser.add_argument('--num-faulty-datapoints', type=int, required=True, help='num faulty datapoins')
    parser.add_argument('--neptune-project', type=str, required=True, help='neptune project name')
    parser.add_argument('--neptune-mode', type=str, required=False, default="debug", help='neptune project name')
    parser.add_argument('--resuming-no', type=int, required=False, default=0, help='the number of combinations for resuming')
    parser.add_argument('--experiment-id', type=str, required=False, default="", help='experiment id')
    parser.add_argument('files', nargs='+', help='metrics files')
    args = parser.parse_args()

    os.environ["TSDR_NEPTUNE_PROJECT"] = args.neptune_project
    os.environ["NEPTUNE_MODE"] = args.neptune_mode

    # Load config python file (.py)
    config_file = os.path.abspath(args.config)
    spec = runpy.run_path(config_file)
    config = spec["CONFIG"]

    logger.info(">>> Loading dataset ...")

    records = loader.load_dataset(
        args.files,
        target_metric_types={
            "containers": True,
            "services": True,
            "nodes": True,
            "middlewares": True,
        },
        num_datapoints=args.num_datapoints,
        validated=True,
        num_faulty_datapoints=args.num_faulty_datapoints,
        max_chaos_case_num=config.pop("max_chaos_case_num"),
    )
    logger.info(f"Load dataset: {len(records)} records")
    assert len(records) > 0, "No records"

    tsdr.sweep_tsdr_and_save_as_cache(records=records, experiment_id=args.experiment_id, resuming_no=args.resuming_no, **config)


if __name__ == "__main__":
    main()
