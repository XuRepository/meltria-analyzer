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
    parser.add_argument('--neptune-mode', type=str, required=True, help='neptune project name')
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
    )
    logger.info(f"Load dataset: {len(records)} records")

    well_injected_records = validation.find_records_detected_anomalies_of_sli(
        records,
        faulty_datapoints=args.num_faulty_datapoints,
    )
    num_sli_anomalies = len(well_injected_records)
    well_injected_records = validation.find_records_detected_anomalies_of_cause_metrics(
        well_injected_records,
        faulty_datapoints=args.num_faulty_datapoints,
        optional_cause=False,
    )
    num_cause_metrics_anomalies = len(well_injected_records)

    logger.info(f"Filtered records: num_sli_anomalies {num_sli_anomalies}, num_cause_metrics_anomalies: {num_cause_metrics_anomalies}")

    tsdr.sweep_tsdr_and_save_as_cache(records=records, **config)


if __name__ == "__main__":
    main()
