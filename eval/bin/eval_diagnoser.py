#!/usr/bin/env python3

import logging
import os
from multiprocessing import cpu_count

import diagnoser.metric_node as mn
import hydra
import meltria.loader as meltria_loader
import neptune.new as neptune
import networkx as nx
import numpy as np
import pandas as pd
from bokeh import plotting
from bokeh.embed import components
from bokeh.resources import CDN
from bs4 import BeautifulSoup
from diagnoser import diag
from eval import groundtruth
from meltria.loader import DatasetRecord
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from pyvis.network import Network
from tsdr import tsdr

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
        G.edges[u, v]["color"] = "black"

    for route in gt_routes:
        node_list = list(route)
        cause_node: mn.MetricNode = node_list[-1]
        G.nodes[cause_node]["color"] = 'red'
        for u, v in zip(node_list, node_list[1:]):
            if G.has_edge(v, u):  # check v -> u
                G.edges[v, u]["color"] = 'red'


def get_html_component_of_time_series_of_graph_nodes(
    record: DatasetRecord, nodes: list[mn.MetricNode], data_df: pd.DataFrame,
) -> tuple[str, str]:
    p: plotting.Figure = plotting.figure(
        title=f"{record.chaos_case_full()}",
        x_axis_label='interval', y_axis_label='zscore',
        width=1000, height=600,
    )
    for node in nodes:
        series: np.ndarray = data_df[node.label].to_numpy()
        if node.is_root():
            p.line(x=np.arange(series.size), y=series, legend_label=f"root metric: {node.label}", line_width=1)
        else:
            p.line(x=np.arange(series.size), y=series, legend_label=f"{node.label}", line_width=1)
    p.add_layout(p.legend[0], 'right')
    script, div = components(p)
    return script, div


def log_causal_graph(
    run: neptune.Run,
    causal_graph: nx.DiGraph,
    record: DatasetRecord,
    gt_routes: list[mn.MetricNodes],
    data_df: pd.DataFrame,
) -> None:
    set_visual_style_to_graph(causal_graph, gt_routes)

    nwg = Network(
        directed=True, height='800px', width='1000px',
        heading=record.chaos_case_full(),
    )
    # pyvis assert isinstance(n_id, str) or isinstance(n_id, int)
    relabeled_mapping = mn.MetricNodes.from_list_of_metric_node(list(causal_graph.nodes)).node_to_label()
    relabeled_graph = nx.relabel_nodes(causal_graph, relabeled_mapping, copy=True)
    nwg.from_nx(relabeled_graph)
    nwg.toggle_physics(True)
    html_path = os.path.join(os.getcwd(), record.basename_of_metrics_file() + '.nw_graph.html')
    nwg.write_html(html_path)

    # append time series plots to the html containing network graph
    with open(html_path) as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        soup.find('head').append(BeautifulSoup(CDN.render_css(), "html.parser")) # load bokehJS
        soup.find('head').append(BeautifulSoup(CDN.render_js(), "html.parser")) # load bokehJS
        bk_script_raw, bk_div_raw = get_html_component_of_time_series_of_graph_nodes(
            record, list(causal_graph.nodes), data_df,
        )
        body = soup.find('body')
        pyvis_div = body.find('div', id="mynetwork")
        bk_div = BeautifulSoup(bk_div_raw, "html.parser").find('div')
        bk_div.attrs['style'] = 'position: relative; top: 800px;'  # place bk canvas below pivis canvas
        pyvis_div.insert_after(bk_div)
        body.append(BeautifulSoup(bk_script_raw, "html.parser"))

    run[f"tests/causal_graphs/{record.chaos_case_full()}"].upload(
        neptune.types.File.from_content(soup.prettify(), extension='html'),
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
                causal_graph, stats = diag.run(
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
            log_causal_graph(run, causal_graph, record, routes, reduced_df)

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
