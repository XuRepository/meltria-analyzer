#!/usr/bin/env python3

import argparse
import logging
import os
import pathlib
import runpy

from eval import localization
from eval.util import logger as internal_logger

DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "dataset" / "data"

internal_logger.logger.setLevel(logging.ERROR)  # Suppress internal logging

# Suppress Neptune log messages to console
npt_logger = logging.getLogger("root_experiment")
npt_logger.setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(prog="sweep_localization")
    parser.add_argument('-c', '--config', nargs="*", type=str, required=True, help='config file')
    parser.add_argument('--neptune-mode', type=str, required=True, help='neptune mode name')
    parser.add_argument('--resuming-no', type=int, required=False, default=0, help='the number of combinations for resuming')
    parser.add_argument('--experiment-id', type=str, required=False, default="", help='experiment id')
    args = parser.parse_args()

    os.environ["NEPTUNE_MODE"] = args.neptune_mode

    # Load config python file (.py)
    for config_file in args.config:
        print(f">> Running {config_file}\n")
        spec = runpy.run_path(config_file)
        config = spec["CONFIG"]
        localization.sweep_localization(**config, experiment_id=args.experiment_id, resuming_no=args.resuming_no)


if __name__ == "__main__":
    main()
