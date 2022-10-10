#!/usr/bin/env python3

import logging
import os
from concurrent import futures
from functools import reduce
from multiprocessing import cpu_count
from operator import add
from typing import Any, cast

import holoviews as hv
import hydra
import neptune.new as neptune
import networkx as nx
import numpy as np
import pandas as pd
from bokeh.embed import file_html
from bokeh.resources import CDN
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from pandas.core.groupby.generic import DataFrameGroupBy

import diagnoser.metric_node as mn
import meltria.loader as meltria_loader
from diagnoser import diag
from eval import groundtruth
from meltria.loader import DatasetRecord
from meltria.priorknowledge.priorknowledge import PriorKnowledge
from tsdr import tsdr

hv.extension("bokeh")

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def set_visual_style_to_graph(G: nx.DiGraph, gt_routes: list[mn.MetricNodes], pk: PriorKnowledge) -> None:
    """Set graph style followed by The valid properties
    https://pyvis.readthedocs.io/en/latest/tutorial.html#adding-list-of-nodes-with-properties
    >>> ['size', 'value', 'title', 'x', 'y', 'label', 'color']
    """
    for node in G.nodes:
        if node.label in pk.get_root_metrics():  # is_root
            color = "orange"
            size = 20
        elif node.is_service():
            color = "blue"
            size = 15
        elif node.is_middleware():
            color = "purple"
            size = 10
        elif node.is_container():
            color = "green"
            size = 10
        else:
            color = "grey"
            size = 10
        G.nodes[node]["color"] = color
        G.nodes[node]["size"] = size
        G.nodes[node]["label"] = node.label
    for u, v in G.edges:
        G.edges[u, v]["line_color"] = "grey"

    for route in gt_routes:
        node_list = list(route)
        cause_node: mn.MetricNode = node_list[-1]
        if G.has_node(cause_node):
            G.nodes[cause_node]["color"] = "red"
        for u, v in zip(node_list, node_list[1:]):
            if G.has_edge(v, u):  # check v -> u
                G.edges[v, u]["line_color"] = "red"


def create_figure_of_causal_graph(
    G: nx.DiGraph,
    record: DatasetRecord,
    width_and_height: tuple[int, int],
):
    """Create a figure of causal graph."""
    opts = dict(
        directed=True,
        tools=["hover", "box_select", "lasso_select", "tap"],
        width=width_and_height[0],
        height=width_and_height[1],
        node_size="size",
        node_color="color",
        cmap=["red", "orange", "blue", "green", "purple", "grey"],
        edge_color="line_color",
        edge_cmap=["red", "grey"],
    )

    # Holoviews Graph only handle a graph whose node type is int or str.
    relabeled_G = mn.relabel_graph_nodes_to_label(G)

    hv_graph = hv.Graph.from_networkx(relabeled_G, nx.layout.spring_layout).opts(
        **opts, title=f"Causal Graph: {record.chaos_case_full()}"
    )
    hv_labels = hv.Labels(hv_graph.nodes, ["x", "y"], "label").opts(
        text_font_size="9pt", text_color="black", bgcolor="white", yoffset=-0.03
    )
    return hv_graph * hv_labels


def create_figure_of_time_series_lines(
    series_df: pd.DataFrame,
    G: nx.DiGraph,
    record: DatasetRecord,
    width_and_height: tuple[int, int],
) -> Any:
    hv_curves = []
    for node in G.nodes:
        series = series_df[node.label]
        df = pd.DataFrame(
            data={
                "x": np.arange(series.size),
                "y": series.to_numpy(),
                "label": node.label,  # to show label with hovertool
            }
        )
        hv_curves.append(hv.Curve(df, label=node.label).opts(tools=["hover", "tap"]))
    return hv.Overlay(hv_curves).opts(
        title=f"Chart of time series metrics {record.chaos_case_full()}",
        tools=["hover", "tap"],
        width=width_and_height[0],
        height=width_and_height[1],
        xlabel="time",
        ylabel="zscore",
        show_grid=True,
        legend_limit=100,
        show_legend=True,
        legend_position="right",
        legend_muted=True,
    )


def hv_render_html(figure, record, suffix) -> str:
    return cast(str, file_html(hv.render(hv.Store.loads(figure)), CDN, f"{record.chaos_case_full()}: {suffix}"))


def log_causal_graph(
    run: neptune.Run,
    causal_subgraphs: tuple[list[nx.DiGraph], list[nx.DiGraph]],
    record: DatasetRecord,
    gt_routes: list[mn.MetricNodes],
    data_df: pd.DataFrame,
) -> None:
    items: list[tuple[list[nx.DiGraph], str, tuple[int, int]]] = [
        (causal_subgraphs[0], "with-root", (1000, 800)),
        (causal_subgraphs[1], "without-root", (600, 400)),
    ]
    with futures.ProcessPoolExecutor(max_workers=len(items)) as executor:
        future_to_suffix: dict[futures.Future, str] = {}
        for (graphs, suffix, (width, height)) in items:
            # Merge small graphs (n <= 2) to speed up the rendering process.
            merged_graphs: list[nx.DiGraph] = [g for g in graphs if g.number_of_nodes() <= 2]
            unmerged_graphs: list[nx.DiGraph] = [g for g in graphs if g not in merged_graphs]
            if len(merged_graphs) > 0:
                unmerged_graphs.append(nx.compose_all(merged_graphs))
            layouts = []
            for graph in unmerged_graphs:
                set_visual_style_to_graph(graph, gt_routes, record.pk)
                nw_graph = create_figure_of_causal_graph(graph, record, (width, height))
                ts_graph = create_figure_of_time_series_lines(data_df, graph, record, (width, height))
                layout = (
                    hv.Layout([nw_graph, ts_graph])
                    .opts(
                        width=width,
                        shared_axes=False,
                    )
                    .cols(1)
                )
                layouts.append(layout)
            figure = (
                reduce(add, layouts)
                .opts(
                    shared_axes=False,
                    title=f"{record.chaos_case_file()}",
                )
                .cols(2)
            )
            # Use hv.store dumps/loads due to https://holoviews.org/FAQ.html
            f = executor.submit(hv_render_html, hv.Store.dumps(figure), record, suffix)
            future_to_suffix[f] = suffix
        for future in futures.as_completed(future_to_suffix):
            suffix = future_to_suffix[future]
            run[f"tests/causal_graphs/{record.chaos_case_full()}-{suffix}"].upload(
                neptune.types.File.from_content(future.result(), extension="html"),
            )


def eval_diagnoser(run: neptune.Run, cfg: DictConfig) -> None:
    dataset_generator = meltria_loader.load_dataset_as_generator(
        cfg.metrics_files,
        cast(dict[str, bool], OmegaConf.to_container(cfg.target_metric_types, resolve=True)),
        cfg.time.num_datapoints,
    )
    logger.info(">> Loading dataset")

    tests_records: list[dict[str, Any]] = []

    for records in dataset_generator:
        for record in records:
            logger.info(f">> Running tsdr {record.chaos_case_file()} ...")

            cfg_step1 = cast(dict[str, Any], OmegaConf.to_container(cfg.tsdr.step1, resolve=True))
            cfg_step2 = cast(dict[str, Any], OmegaConf.to_container(cfg.tsdr.step2, resolve=True))
            tsdr_param = {f"step1_{k}": v for k, v in cfg_step1.items()}
            tsdr_param.update({f"step2_{k}": v for k, v in cfg_step2.items()})
            reducer = tsdr.Tsdr(cfg.tsdr.step1.model_name, **tsdr_param)
            tsdr_stat, _, _ = reducer.run(X=record.data_df, pk=record.pk, max_workers=cpu_count())
            reduced_df: pd.DataFrame = tsdr_stat[-1][0]

            logger.info(f">> Running diagnosis of {record.chaos_case_file()} ...")

            try:
                causal_graph, causal_subgraphs, stats = diag.run(
                    reduced_df,
                    record.pk,
                    **{
                        "pc_library": cfg.params.pc_library,
                        "pc_citest": cfg.params.pc_citest,
                        "pc_citest_alpha": cfg.params.pc_citest_alpha,
                        "pc_variant": cfg.params.pc_variant,
                    },
                )
            except ValueError as e:
                logger.error(e)
                logger.info(f">> Skip because of error {record.chaos_case_file()}")
                continue

            # Check whether cause metrics exists in the causal graph
            _, found_cause_nodes = groundtruth.check_cause_metrics(
                record.pk,
                mn.MetricNodes.from_list_of_metric_node(list(causal_graph.nodes)),
                record.chaos_type(),
                record.chaos_comp(),
            )

            logger.info(f">> Checking causal graph including chaos-injected metrics of {record.chaos_case_file()}")
            graph_ok, routes = groundtruth.check_causal_graph(
                record.pk, causal_graph, record.chaos_type(), record.chaos_comp()
            )
            if not graph_ok:
                logger.info(f"wrong causal graph in {record.chaos_case_file()}")
            tests_records.append(
                {
                    "chaos_type": record.chaos_type(),
                    "chaos_comp": record.chaos_comp(),
                    "metrics_file": record.basename_of_metrics_file(),
                    "graph_ok": graph_ok,
                    "building_graph_elapsed_sec": stats["building_graph_elapsed_sec"],
                    "num_series": tsdr_stat[1][1]["count"].sum(),
                    "init_g_num_nodes": stats["init_graph_nodes_num"],
                    "init_g_num_edges": stats["init_graph_edges_num"],
                    "causal_g_num_nodes": stats["causal_graph_nodes_num"],
                    "causal_g_num_edges": stats["causal_graph_edges_num"],
                    "causal_graph_density": stats["causal_graph_density"],
                    "causal_graph_flow_hierarchy": stats["causal_graph_flow_hierarchy"],
                    "found_routes": ", ".join([route.liststr() for route in routes]),
                    "found_cause_metrics": ", ".join(found_cause_nodes.liststr()),
                    "grafana_dashboard_url": record.grafana_dashboard_url(),
                }
            )
            logger.info(f">> Logging causal graph including chaos-injected metrics of {record.chaos_case_file()}")
            log_causal_graph(run, causal_subgraphs, record, routes, reduced_df)

    tests_df = pd.DataFrame(tests_records).set_index(
        ["chaos_type", "chaos_comp", "metrics_file", "grafana_dashboard_url"]
    )
    run["scores/summary"].upload(neptune.types.File.as_html(tests_df))

    tests_df["accurate"] = np.where(tests_df.graph_ok, 1, 0)
    run["scores"]["tp"] = tests_df["accurate"].agg("sum")
    run["scores"]["accuracy"] = tests_df["accurate"].agg(lambda x: sum(x) / len(x))
    run["scores/building_graph_elapsed_sec"] = tests_df["building_graph_elapsed_sec"].mean()

    def agg_score(df: pd.DataFrame | DataFrameGroupBy) -> pd.DataFrame:
        return df.agg(
            tp=("accurate", "sum"),
            accuracy=("accurate", lambda x: sum(x) / len(x)),
            building_graph_elapsed_sec_mean=("building_graph_elapsed_sec", "mean"),
        )

    run["scores/summary_by_chaos_type"].upload(
        neptune.types.File.as_html(
            agg_score(tests_df.groupby(["chaos_type"])).reset_index(),
        )
    )
    run["scores/summary_by_chaos_comp"].upload(
        neptune.types.File.as_html(
            agg_score(tests_df.groupby(["chaos_comp"])).reset_index(),
        )
    )
    agg_df = agg_score(tests_df.groupby(["chaos_type", "chaos_comp"])).reset_index()
    run["scores/summary_by_chaos_type_and_chaos_comp"].upload(neptune.types.File.as_html(agg_df))
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        logger.info("\n" + agg_df.to_string())


@hydra.main(version_base="1.2", config_path="../conf/diagnoser", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(
        project=os.environ["DIAGNOSER_NEPTUNE_PROJECT"],
        api_token=os.environ["DIAGNOSER_NEPTUNE_API_TOKEN"],
        mode=cfg.neptune.mode,
    )
    npt_handler = NeptuneHandler(run=run)
    logger.addHandler(npt_handler)
    run["dataset/id"] = cfg.dataset_id
    run["dataset/num_metrics_files"] = len(cfg.metrics_files)
    run["parameters"] = {
        "pc_library": cfg.params.pc_library,
        "pc_citest": cfg.params.pc_citest,
        "pc_citest_alpha": cfg.params.pc_citest_alpha,
        "pc_variant": cfg.params.pc_variant,
    }
    run["tsdr/parameters"] = OmegaConf.to_container(cfg.tsdr)
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_diagnoser(run, cfg)

    run.stop()


if __name__ == "__main__":
    main()
