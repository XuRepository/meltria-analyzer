#!/usr/bin/env python3

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../lib")
from metrics import ROOT_METRIC_LABEL, check_cause_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsdrfile",
                        help="tsdr output JSON file")
    args = parser.parse_args()

    with open(args.tsdrfile) as f:
        raw_json = json.load(f)

    metrics: list[str] = raw_json['reduced_metrics']
    method = raw_json['tsdr_method']
    meta = raw_json['metrics_meta']
    chaos_type: str = meta['injected_chaos_type']
    chaos_comp: str = meta['chaos_injected_component']
    root_metrics: list[str] = []
    for column in metrics:
        if column == ROOT_METRIC_LABEL:
            root_metrics.append(column)
    if len(root_metrics) < 1:
        print(
            f"{args.tsdrfile}: [{method}] {chaos_type} to {chaos_comp}: {ROOT_METRIC_LABEL} does not exists")

    ok, cause_metrics = check_cause_metrics(
        metrics, chaos_type, chaos_comp,
    )
    if ok:
        print(
            f"{args.tsdrfile}: [{method}] {chaos_type} to {chaos_comp}: Found cause metrics ({cause_metrics})")
    else:
        print(
            f"{args.tsdrfile}: [{method}] {chaos_type} to {chaos_comp}: Not Found cause metrics")


if __name__ == '__main__':
    main()
