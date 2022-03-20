#!/usr/bin/env python3

import logging
import os
from functools import reduce
from multiprocessing import cpu_count
from operator import add

import diagnoser.metric_node as mn
import holoviews as hv
import hydra
import meltria.loader as meltria_loader
import neptune.new as neptune
import networkx as nx
import numpy as np
import pandas as pd
from bokeh.embed import file_html
from bokeh.models import HoverTool
from bokeh.resources import CDN
from diagnoser import diag
from eval import groundtruth
from meltria.loader import DatasetRecord
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from tsdr import tsdr

hv.extension('bokeh')

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def set_visual_style_to_graph(G: nx.DiGraph, gt_routes: list[mn.MetricNodes]) -> None:
    """Set graph style followed by The valid properties
    https://pyvis.readthedocs.io/en/latest/tutorial.html#adding-list-of-nodes-with-properties
    >>> ['size', 'value', 'title', 'x', 'y', 'label', 'color']
    """
    for node in G.nodes:
        if node.is_root():
            color = "orange"
            size = 25
        elif node.is_service():
            color = "blue"
            size = 20
        elif node.is_middleware():
            color = "purple"
            size = 15
        elif node.is_container():
            color = "green"
            size = 15
        else:
            color = "grey"
            size = 10
        G.nodes[node]["color"] = color
        G.nodes[node]["size"] = size
        G.nodes[node]["label"] = node.label
    for u, v in G.edges:
        G.edges[u, v]["color"] = "black"

    for route in gt_routes:
        node_list = list(route)
        cause_node: mn.MetricNode = node_list[-1]
        if G.has_node(cause_node):
            G.nodes[cause_node]["color"] = 'red'
        for u, v in zip(node_list, node_list[1:]):
            if G.has_edge(v, u):  # check v -> u
                G.edges[v, u]["color"] = 'red'


def create_figure_of_causal_graph(
    root_contained_graphs: list[nx.DiGraph],
    root_uncontained_graphs: list[nx.DiGraph],
    record: DatasetRecord,
):
    """ Create a figure of causal graph.
    """
    opts = dict(
        directed=True,
        tools=['hover', 'box_select', 'lasso_select', 'tap'],
        width=800, height=600,
        node_size='size', node_color='color',
        cmap=['red', 'orange', 'blue', 'green', 'purple', 'grey'],
        edge_color='color', edge_cmap=['red', 'black'],
    )

    def create_graph(G: nx.DiGraph, title: str):
        hv_graph = hv.Graph.from_networkx(G, nx.layout.kamada_kawai_layout).opts(
            **opts, title=title)
        hv_labels = hv.Labels(hv_graph.nodes, ['x', 'y'], 'label').opts(
            text_font_size='10pt', text_color='black', bgcolor='white', yoffset=-0.06)
        return (hv_graph * hv_labels)

    return reduce(
        add,
        [create_graph(g, f"Causal Graph with root: {record.chaos_case_full()}") for g in root_contained_graphs] + \
        [create_graph(g, f"Causal Graph without root: {record.chaos_case_full()}") for g in root_uncontained_graphs]
    )


def create_figure_of_time_series_lines(
    series_df: pd.DataFrame,
    root_contained_graphs: list[nx.DiGraph],
    root_uncontained_graphs: list[nx.DiGraph],
    record: DatasetRecord,
):
    hv_curves = []
    hover = HoverTool(description='Custom Tooltip', tooltips=[("(x,y)", "($x, $y)"), ('label', '@label')])
    for G in root_contained_graphs + root_uncontained_graphs:
        for node in G.nodes:
            series = series_df[node.label]
            df = pd.DataFrame(data={
                'x': np.arange(series.size),
                'y': series.to_numpy(),
                'label': node.label,  # to show label with hovertool
            })
            if node.is_root():
                c = hv.Curve(df, label=node.label, group='root').opts(tools=[hover, 'tap'])
            else:
                c = hv.Curve(df, label=node.label).opts(tools=[hover, 'tap'])
            hv_curves.append(c)
    return hv.Overlay(hv_curves).opts(
        tools=['hover', 'tap'],
        height=400,
        width=1000,
        xlabel='time',
        ylabel='zscore',
        show_grid=True,
        title=f'Chart of time series metrics {record.chaos_case_full()}',
        legend_position='right',
        legend_muted=True,
    )


def log_causal_graph(
    run: neptune.Run,
    causal_subgraphs: tuple[list[nx.DiGraph], list[nx.DiGraph]],
    record: DatasetRecord,
    gt_routes: list[mn.MetricNodes],
    data_df: pd.DataFrame,
) -> None:
    for graphs in causal_subgraphs:
        for g in graphs:
            set_visual_style_to_graph(g, gt_routes)

    # Holoviews only handle a graph whose node type is int or str.
    relabeled_subgraphs = tuple([mn.relabel_graph_nodes_to_label(g) for g in graphs] for graphs in causal_subgraphs)

    hv_graph_with_labels = create_figure_of_causal_graph(relabeled_subgraphs[0], relabeled_subgraphs[1], record)
    ts_graph = create_figure_of_time_series_lines(data_df, causal_subgraphs[0], causal_subgraphs[1], record)
    layout = hv.Layout([hv_graph_with_labels, ts_graph]).opts(
        shared_axes=False, width=1200,
        title=f"{record.chaos_case_file()}",
    ).cols(1)

    html = file_html(hv.render(layout), CDN, f"{record.chaos_case_full()}")
    run[f"tests/causal_graphs/{record.chaos_case_full()}"].upload(
        neptune.types.File.from_content(html, extension='html'),
    )


def eval_diagnoser(run: neptune.Run, cfg: DictConfig) -> None:
    dataset, mappings_by_metrics_file = meltria_loader.load_dataset(
        cfg.metrics_files,
        cfg.exclude_middleware_metrics,
    )
    logger.info("Dataset loading complete.")

    tests_df = pd.DataFrame(
        columns=[
            'chaos_type', 'chaos_comp', 'metrics_file', 'graph_ok', 'building_graph_elapsed_sec',
            'num_series', 'init_g_num_nodes', 'init_g_num_edges', 'g_num_nodes', 'g_num_edges', 'g_density',
            'g_flow_hierarchy', 'found_routes', 'found_cause_metrics', 'grafana_dashboard_url',
        ],
        index=['chaos_type', 'chaos_comp', 'metrics_file', 'grafana_dashboard_url'],
    ).dropna()

    for (chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1]):
        y_pred: list[int] = []
        graph_building_elapsed_secs: list[float] = []

        for (metrics_file, grafana_dashboard_url), data_df in sub_df.groupby(level=[2, 3]):
            record = DatasetRecord(chaos_type, chaos_comp, metrics_file, data_df)

            logger.info(f">> Running tsdr {record.chaos_case_file()} ...")

            reducer = tsdr.Tsdr(tsdr.ar_based_ad_model, **{
                'tsifter_step1_ar_regression': cfg.tsdr.step1.ar_regression,
                'tsifter_step1_ar_anomaly_score_threshold': cfg.tsdr.step1.ar_anomaly_score_threshold,
                'tsifter_step1_cv_threshold': cfg.tsdr.step1.cv_threshold,
                'tsifter_step1_ar_dynamic_prediction': cfg.tsdr.step1.ar_dynamic_prediction,
                'tsifter_step2_clustering_threshold': cfg.tsdr.step2.dist_threshold,
                'tsifter_step2_clustered_series_type': cfg.tsdr.step2.clustered_series_type,
                'tsifter_step2_clustering_dist_type': cfg.tsdr.step2.clustering_dist_type,
                'tsifter_step2_clustering_choice_method': cfg.tsdr.step2.clustering_choice_method,
                'tsifter_step2_clustering_linkage_method': cfg.tsdr.step2.clustering_linkage_method,
            })
            _, reduced_df_by_step, metrics_dimension, _ = reducer.run(
                series=data_df,
                max_workers=cpu_count(),
            )
            reduced_df: pd.DataFrame = reduced_df_by_step['step2']

            logger.info(f">> Running diagnosis of {record.chaos_case_file()} ...")

            try:
                causal_graph, causal_subgraphs, stats = diag.run(
                    reduced_df, mappings_by_metrics_file[record.metrics_file], **{
                        'pc_library': cfg.params.pc_library,
                        'pc_citest': cfg.params.pc_citest,
                        'pc_citest_alpha': cfg.params.pc_citest_alpha,
                        'pc_variant': cfg.params.pc_variant,
                    }
                )
            except ValueError as e:
                logger.error(e)
                logger.info(f">> Skip because of error {record.chaos_case_file()}")
                continue

            # Check whether cause metrics exists in the causal graph
            _, found_cause_nodes = groundtruth.check_cause_metrics(
                mn.MetricNodes.from_list_of_metric_node(list(causal_graph.nodes)), chaos_type, chaos_comp,
            )

            logger.info(f">> Checking causal graph including chaos-injected metrics of {record.chaos_case_file()}")
            graph_ok, routes = groundtruth.check_causal_graph(causal_graph, chaos_type, chaos_comp)
            if not graph_ok:
                logger.info(f"wrong causal graph in {record.chaos_case_file()}")
            graph_building_elapsed_secs.append(stats['building_graph_elapsed_sec'])
            tests_df = tests_df.append(
                pd.Series(
                    [
                        chaos_type, chaos_comp, metrics_file, graph_ok, stats['building_graph_elapsed_sec'],
                        metrics_dimension['total'][2],
                        stats['init_graph_nodes_num'], stats['init_graph_edges_num'],
                        stats['causal_graph_nodes_num'], stats['causal_graph_edges_num'],
                        stats['causal_graph_density'], stats['causal_graph_flow_hierarchy'],
                        ', '.join([route.liststr() for route in routes]),
                        found_cause_nodes.liststr(), grafana_dashboard_url,
                    ], index=tests_df.columns,
                ), ignore_index=True,
            )
            log_causal_graph(run, causal_subgraphs, record, routes, reduced_df)

    tests_df['accurate'] = np.where(tests_df.graph_ok, 1, 0)
    run['scores']['tp'] = tests_df['accurate'].agg('sum')
    run['scores']['accuracy'] = tests_df['accurate'].agg(lambda x: sum(x) / len(x))
    run['scores/building_graph_elapsed_sec'] = tests_df['building_graph_elapsed_sec'].mean()

    def agg_score(df) -> pd.DataFrame:
        return df.agg(
            tp=('accurate', 'sum'),
            accuracy=('accurate', lambda x: sum(x) / len(x)),
            building_graph_elapsed_sec_mean=('building_graph_elapsed_sec', 'mean'),
        )

    run['scores/summary_by_chaos_type'].upload(neptune.types.File.as_html(
        agg_score(tests_df.groupby(['chaos_type'])).reset_index(),
    ))
    run['scores/summary_by_chaos_comp'].upload(neptune.types.File.as_html(
        agg_score(tests_df.groupby(['chaos_comp'])).reset_index(),
    ))
    agg_df = agg_score(tests_df.groupby(['chaos_type', 'chaos_comp'])).reset_index()
    run['scores/summary_by_chaos_type_and_chaos_comp'].upload(neptune.types.File.as_html(agg_df))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n"+agg_df.to_string())


@hydra.main(config_path='../conf/diagnoser', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(
        project=os.environ['DIAGNOSER_NEPTUNE_PROJECT'],
        api_token=os.environ['DIAGNOSER_NEPTUNE_API_TOKEN'],
        mode=cfg.neptune.mode,
    )
    npt_handler = NeptuneHandler(run=run)
    logger.addHandler(npt_handler)
    run['dataset/id'] = cfg.dataset_id
    run['dataset/num_metrics_files'] = len(cfg.metrics_files)
    run['parameters'] = {
        'pc_library': cfg.params.pc_library,
        'pc_citest': cfg.params.pc_citest,
        'pc_citest_alpha': cfg.params.pc_citest_alpha,
        'pc_variant': cfg.params.pc_variant,
    }
    run['tsdr/parameters'] = OmegaConf.to_container(cfg.tsdr)
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_diagnoser(run, cfg)

    run.stop()


if __name__ == '__main__':
    main()
