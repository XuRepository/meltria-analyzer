#!/usr/bin/env python3

import logging
import os
from concurrent import futures
from dataclasses import dataclass
from functools import reduce
from multiprocessing import cpu_count
from operator import add
from typing import Any, cast

import holoviews as hv
import hydra
import neptune.new as neptune
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from bokeh.embed import file_html
from bokeh.resources import CDN
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, precision_score, recall_score

import meltria.loader as meltria_loader
from eval import groundtruth
from eval.validation import check_valid_dataset, detect_with_n_sigma_rule
from meltria.loader import DatasetRecord
from tsdr import tsdr

hv.extension("bokeh")  # type: ignore


# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger("root_experiment")
logger.setLevel(logging.INFO)


@dataclass
class TimeSeriesPlotter:
    run: neptune.Run
    enable_upload_plots: bool
    logger: logging.Logger

    def log_plots_as_html(self, record: DatasetRecord) -> None:
        """Upload found_metrics plot images to neptune.ai."""
        if not self.enable_upload_plots:
            return
        self.logger.info(f">> Uploading plot figures of {record.chaos_case_file()} ...")
        if (gtdf := record.ground_truth_metrics_frame()) is None:
            return
        html = self.generate_html_time_series(
            record,
            gtdf,
            title=f"Chart of time series metrics {record.chaos_case_full()}",
        )
        self.run[f"dataset/figures/{record.chaos_case_full()}"].upload(
            neptune.types.File.from_content(html, extension="html"),
        )

    def log_clustering_plots_as_html(
        self,
        clustering_info: dict[str, list[str]],
        non_clustered_reduced_df: pd.DataFrame,
        record: DatasetRecord,
        anomaly_points: dict[str, np.ndarray],
    ) -> None:
        """Upload clustered time series plots to neptune.ai."""
        if not self.enable_upload_plots:
            return

        # Parallelize plotting of clustered and no-clustered metrics.
        with futures.ProcessPoolExecutor(max_workers=2) as executor:
            future_list: dict[futures.Future, str] = {}
            f = executor.submit(
                self.get_html_of_clustered_series_plots,
                clustering_info=clustering_info,
                record=record,
                anomaly_points=anomaly_points,
            )
            future_list[f] = f"tests/clustering/time_series_plots/{record.chaos_case_full()}.clustered"
            f = executor.submit(
                self.get_html_of_non_clustered_series_plots,
                non_clustered_reduced_df=non_clustered_reduced_df,
                record=record,
                anomaly_points=anomaly_points,
            )
            future_list[f] = f"tests/clustering/time_series_plots/{record.chaos_case_full()}.no_clustered"
            for future in futures.as_completed(future_list):
                neptune_path: str = future_list[future]
                html: str = future.result()
                self.run[neptune_path].upload(
                    neptune.types.File.from_content(html, extension="html"),
                )

    @classmethod
    def get_html_of_clustered_series_plots(
        cls,
        clustering_info: dict[str, list[str]],
        record: DatasetRecord,
        anomaly_points: dict[str, np.ndarray],
    ) -> str:
        """Upload clustered time series plots to neptune.ai."""
        logger.info(f">> Uploading clustering plots of {record.chaos_case_file()} ...")
        figures: list[hv.Overlay] = []
        for rep_metric, sub_metrics in clustering_info.items():
            clustered_metrics: list[str] = [rep_metric] + sub_metrics
            fig: hv.Overlay = cls.generate_figure_time_series(
                data=record.data_df[clustered_metrics],
                anomaly_points=anomaly_points,
                title=f"Chart of time series metrics {record.chaos_case_full()} / rep:{rep_metric}",
                width_and_height=(800, 400),
            )
            figures.append(fig)
        if len(figures) == 0:
            return ""
        final_fig = reduce(add, figures)
        return cast(str, file_html(hv.render(final_fig), CDN, record.chaos_case_full()))

    @classmethod
    def get_html_of_non_clustered_series_plots(
        cls,
        record: DatasetRecord,
        non_clustered_reduced_df: pd.DataFrame,
        anomaly_points: dict[str, np.ndarray],
    ) -> str:
        """Upload non-clustered time series plots to neptune.ai."""
        logger.info(f">> Uploading non-clustered plots of {record.chaos_case_file()} ...")
        if len(non_clustered_reduced_df.columns) == 0:
            return ""

        figures: list[hv.Overlay] = []
        for service, metrics in record.pk.group_metrics_by_service(list(non_clustered_reduced_df.columns)).items():
            fig: hv.Overlay = cls.generate_figure_time_series(
                data=non_clustered_reduced_df[metrics],
                anomaly_points=anomaly_points,
                title=f"Chart of time series metrics {record.chaos_case_full()} / {service} [no clustered]",
                width_and_height=(800, 400),
            )
            figures.append(fig)
        final_fig = reduce(add, figures)
        return cast(str, file_html(hv.render(final_fig), CDN, record.chaos_case_full()))

    @classmethod
    def generate_html_time_series(
        cls,
        record: DatasetRecord,
        data: pd.DataFrame,
        title: str,
        anomaly_points: dict[str, np.ndarray] = {},
    ) -> str:
        fig = cls.generate_figure_time_series(data, title=title, anomaly_points=anomaly_points)
        return cast(str, file_html(hv.render(fig), CDN, record.chaos_case_full()))

    @classmethod
    def generate_figure_time_series(
        cls,
        data: pd.DataFrame,
        title: str,
        width_and_height: tuple[int, int] = (1200, 600),
        anomaly_points: dict[str, np.ndarray] = {},
    ) -> hv.Overlay:
        hv_curves = []
        for column in data.columns:
            vals: np.ndarray = scipy.stats.zscore(data[column].to_numpy())
            df = pd.DataFrame(
                data={
                    "x": np.arange(vals.size),
                    "y": vals,
                    "label": column,  # to show label with hovertool
                }
            )
            line = hv.Curve(df, label=column).opts(tools=["hover", "tap"])
            if (points := anomaly_points.get(column)) is None:
                hv_curves.append(line)
            else:
                ap = np.array([(p[0], vals[p[0]]) for p in points])
                hv_curves.append(line * hv.Points(ap).opts(color="red", size=8, marker="x"))
        return hv.Overlay(hv_curves).opts(  # type: ignore
            title=title,
            tools=["hover", "tap"],
            width=width_and_height[0],
            height=width_and_height[1],
            xlabel="time",
            ylabel="zscore",
            fontsize={"legend": 8},
            show_grid=True,
            legend_limit=100,
            show_legend=True,
            legend_position="right",
            legend_muted=True,
        )


def prepare_ground_truth_labels(data_df: pd.DataFrame, labbeling: dict[str, Any], fi_time: int) -> pd.DataFrame:
    """Prepare ground truth labels for anomaly detection."""
    gt_labels: dict[int, pd.Series] = {}
    for n_sigma in labbeling["n_sigma_rule"]["n_sigmas"]:
        gt_labels[n_sigma] = data_df.apply(
            lambda X: detect_with_n_sigma_rule(X, test_start_time=fi_time, sigma_threshold=n_sigma).size > 0
        ).astype(bool)
    return pd.DataFrame.from_dict(gt_labels, orient="index")


def calculate_performance_metrics_based_labeling(
    ground_truth_labels: pd.DataFrame,
    tsdr_data_df: pd.DataFrame,  # data after reduction
) -> pd.DataFrame:
    """Calculate accuracy of anomaly points."""

    scores: list[tuple[int, float, float, float]] = []
    tsdr_data_df = tsdr_data_df.sort_index(axis=1)
    gt_labels = ground_truth_labels.sort_index(axis=1)
    full_cols: list[str] = gt_labels.columns.to_list()
    for n_sigma, row in gt_labels.iterrows():
        y_true: np.ndarray = row.to_numpy()
        y_pred: np.ndarray = np.full_like(y_true, fill_value=False)
        last_pos: int = 0
        for col in tsdr_data_df.columns:
            if (pos := full_cols[last_pos:].index(col)) == -1:
                raise ValueError(f"Column {col} not found in ground truth")
            y_pred[last_pos + pos] = True
            last_pos = pos

        scores.append(
            (
                int(cast(int, n_sigma)),
                cast(float, accuracy_score(y_true, y_pred)),
                cast(float, precision_score(y_true, y_pred)),
                cast(float, recall_score(y_true, y_pred)),
            )
        )
    return pd.DataFrame(scores, columns=["n_sigma", "accuracy", "precision", "recall"]).set_index(["n_sigma"])


def eval_tsdr_a_record(
    record: DatasetRecord,
    cfg: DictConfig,
    ts_plotter: TimeSeriesPlotter,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, str]], dict[str, str]]:
    labbeling = cast(dict[str, Any], OmegaConf.to_container(cfg.labbeling, resolve=True))
    fault_inject_time_index = cast(int, cfg.time.fault_inject_time_index)
    # check any True of all causal paths
    valid_dataset_ok: bool | None = None
    if cfg.disable_dataset_validation:
        logger.info(f">> Skip validation of dataset {record.chaos_case_full()} ...")
    else:
        logger.info(f">> Validating dataset {record.chaos_case_full()} ...")
        valid_dataset_ok = check_valid_dataset(record, labbeling, fault_inject_time_index)

    ts_plotter.log_plots_as_html(record)

    logger.info(f">> Running tsdr {record.chaos_case_full()} ...")

    tsdr_param = {"time_fault_inject_time_index": cfg.time.fault_inject_time_index}
    pycfg_step1 = cast(dict[str, Any], OmegaConf.to_container(cfg.step1, resolve=True))
    pycfg_step2 = cast(dict[str, Any], OmegaConf.to_container(cfg.step2, resolve=True))
    tsdr_param.update({f"step1_{k}": v for k, v in pycfg_step1.items()})
    tsdr_param.update({f"step2_{k}": v for k, v in pycfg_step2.items()})
    reducer = tsdr.Tsdr(cfg.step1.model_name, **tsdr_param)
    tsdr_stat, clustering_info, anomaly_points = reducer.run(
        X=record.data_df,
        pk=record.pk,
        max_workers=cpu_count(),
    )

    logger.info(f">> Evaluating tsdr {record.chaos_case_full()} ...")

    filtered_df: pd.DataFrame = tsdr_stat[1][0]  # simple filtered-out data
    ground_truth_labels: pd.DataFrame = prepare_ground_truth_labels(filtered_df, labbeling, fault_inject_time_index)
    tests_items: list[dict[str, Any]] = []
    perf_metrics_dfs: list[pd.DataFrame] = []
    # skip the first item of tsdr_stat because it
    for i, (reduced_df, stat_df, elapsed_time) in enumerate(tsdr_stat[1:], start=1):
        ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
            pk=record.pk,
            metrics=list(reduced_df.columns),
            chaos_type=record.chaos_type(),
            chaos_comp=record.chaos_comp(),
        )
        num_series_by_type: dict[str, int] = {}
        for metric_type, ok in cfg.target_metric_types.items():
            if not ok:
                continue
            num_series_by_type[f"num_series/{metric_type}/raw"] = tsdr_stat[0][1].loc[metric_type]["count"].sum()
            num_series_by_type[f"num_series/{metric_type}/filtered"] = tsdr_stat[1][1].loc[metric_type]["count"].sum()
            num_series_by_type[f"num_series/{metric_type}/reduced"] = stat_df.loc[metric_type]["count"].sum()
        tests_items.append(
            {
                "chaos_type": record.chaos_type(),
                "chaos_comp": record.chaos_comp(),
                "metrics_file": record.basename_of_metrics_file(),
                "step": f"step{i}",
                "valid_dataset_ok": valid_dataset_ok,
                "ok": ok,
                "num_series/total/raw": tsdr_stat[0][1]["count"].sum(),  # raw
                "num_series/total/filtered": tsdr_stat[1][1]["count"].sum(),  # after step0
                "num_series/total/reduced": stat_df["count"].sum(),  # after step{i}
                **num_series_by_type,
                "elapsed_time": elapsed_time,
                "found_metrics": ",".join(found_metrics),
                "grafana_dashboard_url": record.grafana_dashboard_url(),
            }
        )

        perf_metrics_df = calculate_performance_metrics_based_labeling(ground_truth_labels, reduced_df)
        perf_metrics_df["chaos_type"] = record.chaos_type()
        perf_metrics_df["chaos_comp"] = record.chaos_comp()
        perf_metrics_df["metrics_file"] = record.basename_of_metrics_file()
        perf_metrics_df["step"] = f"step{i}"
        perf_metrics_dfs.append(perf_metrics_df)

    clustering_items: list[dict[str, str]] = []
    for representative_metric, sub_metrics in clustering_info.items():
        clustering_items.append(
            {
                "chaos_type": record.chaos_type(),
                "chaos_comp": record.chaos_comp(),
                "metrics_file": record.basename_of_metrics_file(),
                "representative_metric": representative_metric,
                "sub_metrics": ",".join(sub_metrics),
            }
        )

    rep_metrics: list[str] = list(clustering_info.keys())
    post_clustered_reduced_df = tsdr_stat[-1][0]  # the last item pf tsdr_stat should be clustered result.
    non_clustered_reduced_df: pd.DataFrame = post_clustered_reduced_df.drop(columns=rep_metrics)
    non_clustered_item: dict[str, str] = {
        "chaos_type": record.chaos_type(),
        "chaos_comp": record.chaos_comp(),
        "metrics_file": record.basename_of_metrics_file(),
        "non_clustered_metrics": ",".join(non_clustered_reduced_df.columns),
    }

    ts_plotter.log_clustering_plots_as_html(
        clustering_info,
        non_clustered_reduced_df,
        record,
        anomaly_points,
    )

    return tests_items, pd.concat(perf_metrics_dfs, copy=False), clustering_items, non_clustered_item


def save_scores(
    run: neptune.Run,
    tests: list[dict[str, Any]],
    perf_metrics_df: pd.DataFrame,
    clustering: list[dict[str, Any]],
    non_clustered: list[dict[str, Any]],
    target_metric_types: dict[str, bool],
) -> None:
    clustering_df = pd.DataFrame(clustering).set_index(
        [
            "chaos_type",
            "chaos_comp",
            "metrics_file",
            "representative_metric",
            "sub_metrics",
        ]
    )
    non_clustered_df = pd.DataFrame(non_clustered).set_index(["chaos_type", "chaos_comp", "metrics_file"])
    tests_df = pd.DataFrame(tests).set_index(
        ["chaos_type", "chaos_comp", "metrics_file", "grafana_dashboard_url", "step"]
    )

    run["tests/clustering/clustered_table"].upload(neptune.types.File.as_html(clustering_df))
    run["tests/clustering/non_clustered_table"].upload(neptune.types.File.as_html(non_clustered_df))

    run["scores/summary"].upload(neptune.types.File.as_html(tests_df))

    def agg_score(x: pd.DataFrame) -> pd.Series:
        tp = int(x["ok"].sum())
        fn = int((~x["ok"]).sum())
        rate = 1 - x["num_series/total/reduced"] / x["num_series/total/filtered"]
        valid: pd.Series = x["valid_dataset_ok"]
        num_series_items: dict[str, str] = {}
        for metric_type, ok in target_metric_types.items():
            if not ok:
                continue
            num_series_items[f"num_series/{metric_type}"] = "/".join(
                [
                    f"{int(x[f'num_series/{metric_type}/reduced'].mean())}",
                    f"{int(x[f'num_series/{metric_type}/filtered'].mean())}",
                    f"{int(x[f'num_series/{metric_type}/raw'].mean())}",
                ]
            )
        d = {
            "data_validity": int(valid.sum()) / valid.size if valid.notnull().any() else np.NaN,
            "tp": tp,
            "fn": fn,
            "accuracy": tp / (tp + fn),
            "num_series/total": "/".join(
                [
                    f"{int(x['num_series/total/reduced'].mean())}",
                    f"{int(x['num_series/total/filtered'].mean())}",
                    f"{int(x['num_series/total/raw'].mean())}",
                ]
            ),
            **num_series_items,
            "reduction_rate_mean": rate.mean(),
            "reduction_rate_max": rate.max(),
            "reduction_rate_min": rate.min(),
            "elapsed_time": x["elapsed_time"].mean(),
            "elapsed_time_max": x["elapsed_time"].max(),
            "elapsed_time_min": x["elapsed_time"].min(),
        }
        return pd.Series(d)

    scores_by_step = tests_df.groupby("step").apply(agg_score).reset_index().set_index("step")
    scores_by_chaos_type = (
        tests_df.groupby(["chaos_type", "step"]).apply(agg_score).reset_index().set_index(["chaos_type", "step"])
    )
    scores_by_chaos_comp = (
        tests_df.groupby(["chaos_comp", "step"]).apply(agg_score).reset_index().set_index(["chaos_comp", "step"])
    )
    scores_by_chaos_type_and_comp = (
        tests_df.groupby(
            ["chaos_type", "chaos_comp", "step"],
        )
        .apply(agg_score)
        .reset_index()
        .set_index(["chaos_type", "chaos_comp", "step"])
    )
    total_scores: pd.Series = scores_by_step.loc["step2"]
    for col in ["elapsed_time", "elapsed_time_max", "elapsed_time_min"]:
        total_scores[col] = scores_by_step.loc["step1"][col] + scores_by_step.loc["step2"][col]
    perf_metrics = (
        perf_metrics_df.groupby(["chaos_type", "chaos_comp", "step", "n_sigma"])
        .agg(["mean", "max", "min"])
        .reset_index()
        .set_index(["chaos_type", "chaos_comp", "step", "n_sigma"])
    )

    run["scores"] = total_scores.to_dict()
    run["scores/summary_by_step"].upload(neptune.types.File.as_html(scores_by_step))
    run["scores/summary_by_chaos_type"].upload(neptune.types.File.as_html(scores_by_chaos_type))
    run["scores/summary_by_chaos_comp"].upload(neptune.types.File.as_html(scores_by_chaos_comp))
    run["scores/summary_by_chaos_type_and_chaos_comp"].upload(
        neptune.types.File.as_html(scores_by_chaos_type_and_comp)
    )
    run["scores/perf_metrics"].upload(neptune.types.File.as_html(perf_metrics))
    for df in [
        scores_by_step,
        scores_by_chaos_type,
        scores_by_chaos_comp,
        scores_by_chaos_type_and_comp,
        perf_metrics,
    ]:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):  # type: ignore
            logger.info("\n" + df.to_string())


def eval_tsdr(run: neptune.Run, cfg: DictConfig) -> None:
    ts_plotter: TimeSeriesPlotter = TimeSeriesPlotter(
        run=run,
        enable_upload_plots=cfg.upload_plots,
        logger=logger,
    )

    dataset_generator = meltria_loader.load_dataset_as_generator(
        cfg.metrics_files,
        cast(dict[str, bool], OmegaConf.to_container(cfg.target_metric_types, resolve=True)),
        cfg.time.num_datapoints,
    )
    logger.info("Loading metrics files")

    clustering_items: list[dict[str, Any]] = []
    performance_metrics_dfs: list[pd.DataFrame] = []
    non_clustered_items: list[dict[str, Any]] = []
    tests_items: list[dict[str, Any]] = []

    for records in dataset_generator:
        for record in records:
            _tests_items, _performance_metrics_df, _clustering_items, _non_clustered_item = eval_tsdr_a_record(
                record, cfg, ts_plotter
            )
            tests_items.extend(_tests_items)
            clustering_items.extend(_clustering_items)
            non_clustered_items.append(_non_clustered_item)
            performance_metrics_dfs.append(_performance_metrics_df)

            del record  # reduce memory usage

    save_scores(
        run,
        tests_items,
        pd.concat(performance_metrics_dfs, copy=False),
        clustering_items,
        non_clustered_items,
        cfg.target_metric_types,
    )


@hydra.main(version_base="1.2", config_path="../conf/tsdr", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(
        project=os.environ["TSDR_NEPTUNE_PROJECT"],
        api_token=os.environ["TSDR_NEPTUNE_API_TOKEN"],
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
    pycfg_step1 = cast(dict[str, Any], OmegaConf.to_container(cfg.step1, resolve=True))
    params.update({f"step1_{k}": v for k, v in pycfg_step1.items()})
    pycfg_step2 = cast(dict[str, Any], OmegaConf.to_container(cfg.step2, resolve=True))
    params.update({f"step2_{k}": v for k, v in pycfg_step2.items()})

    run["parameters"] = params
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_tsdr(run, cfg)

    run.stop()


if __name__ == "__main__":
    main()
