#!/usr/bin/env python3

import argparse
import json

from eval.groundtruth import check_cause_metrics
from meltria.priorknowledge.priorknowledge import new_knowledge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsdrfiles", nargs="+", help="tsdr output JSON file")
    args = parser.parse_args()

    for tsdrfile in args.tsdrfiles:
        with open(tsdrfile) as f:
            raw_json = json.load(f)

        metrics: list[str] = raw_json["reduced_metrics"]
        method = raw_json["tsdr_method"]
        meta = raw_json["metrics_meta"]
        pk = new_knowledge(meta["target_app"])
        chaos_type: str = meta["injected_chaos_type"]
        chaos_comp: str = meta["chaos_injected_component"]
        root_metrics: list[str] = []
        for column in metrics:
            if column in pk.get_root_metrics():
                root_metrics.append(column)
        if len(root_metrics) < 1:
            print(f"{tsdrfile}: [{method}] {chaos_type} to {chaos_comp}: {pk.get_root_metrics()} does not exists")

        ok, cause_metrics = check_cause_metrics(
            metrics,
            chaos_type,
            chaos_comp,
        )
        if ok:
            print(f"{tsdrfile}: [{method}] {chaos_type} to {chaos_comp}: Found cause metrics ({cause_metrics})")
        else:
            print(f"{tsdrfile}: [{method}] {chaos_type} to {chaos_comp}: Not Found cause metrics")


if __name__ == "__main__":
    main()
