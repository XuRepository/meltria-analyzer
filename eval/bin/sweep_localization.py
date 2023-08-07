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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)

# Suppress Neptune log messages to console
npt_logger = logging.getLogger("root_experiment")
npt_logger.setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(prog="sweep_localization")
    parser.print_usage = parser.print_help
    parser.add_argument('-c', '--config', type=str, required=True, help='config file')
    parser.add_argument('--neptune-mode', type=str, required=True, help='neptune mode name')
    args = parser.parse_args()

    os.environ["NEPTUNE_MODE"] = args.neptune_mode

    # Load config python file (.py)
    config_file = os.path.abspath(args.config)
    spec = runpy.run_path(config_file)
    config = spec["CONFIG"]

    localization.sweep_localization(**config)


if __name__ == "__main__":
    main()
