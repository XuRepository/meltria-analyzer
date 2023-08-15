import sys
import warnings
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.append((Path(__file__).parent.parent / "PyRCA").as_posix())

import circa.alg.random_walk as rw
from pyrca.analyzers.epsilon_diagnosis import (EpsilonDiagnosis,
                                               EpsilonDiagnosisConfig)
from pyrca.analyzers.ht import HT, HTConfig
from pyrca.analyzers.random_walk import RandomWalk, RandomWalkConfig
from pyrca.analyzers.rcd import RCD, RCDConfig
from pyrca.graphs.causal.base import CausalModel
from pyrca.graphs.causal.fges import FGES, FGESConfig
from pyrca.graphs.causal.ges import GES, GESConfig
from pyrca.graphs.causal.lingam import LiNGAM, LiNGAMConfig
from pyrca.graphs.causal.pc import PC, PCConfig

import diagnoser.metric_node as mn
from diagnoser import call_graph
from meltria.priorknowledge.priorknowledge import PriorKnowledge


def run_localization(
    dataset: pd.DataFrame,
    pk: PriorKnowledge,
    method: str,
    enable_prior_knowledge: bool = False,
    walk_method: str | None = None,
    root_cause_top_k: int = 25,
    anomalous_metrics: list[str] | None = None,
    **kwargs,
) -> list[tuple[str, float]]:
    if "pyrca_boundary_index" in kwargs:
        normal_df = dataset[dataset.index < kwargs["pyrca_boundary_index"]]
        anomalous_df = dataset[dataset.index >= kwargs["pyrca_boundary_index"]]
    else:
        normal_df, anomalous_df = None, None

    graph: pd.DataFrame
    match method:
        case "epsilon_diagnosis":
            assert normal_df is not None and anomalous_df is not None
            # e-Diagnosis requires the size of normal_df and anomalous_df to be the same
            normal_df = normal_df.iloc[-len(anomalous_df):, :]
            model = EpsilonDiagnosis(config=EpsilonDiagnosisConfig(root_cause_top_k=root_cause_top_k))
            model.train(normal_df)
            results = model.find_root_causes(anomalous_df).to_list()
            return [(r["root_cause"], r["score"]) for i, r in enumerate(results)]
        case "rcd":
            assert normal_df is not None and anomalous_df is not None
            model = RCD(config=RCDConfig(k=root_cause_top_k, localized=True))
            results = model.find_root_causes(normal_df, anomalous_df).to_list()
            return [(r["root_cause"], r["score"]) for i, r in enumerate(results)]
        case "pc":
            forbits = call_graph.get_forbits(dataset, pk) if enable_prior_knowledge else []
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                graph = PC(PCConfig(run_pdag2dag=True)).train(dataset, forbits=forbits)
        case "lingam":
            forbits = call_graph.get_forbits(dataset, pk) if enable_prior_knowledge else []
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                graph = LiNGAM(LiNGAMConfig(run_pdag2dag=True)).train(dataset, forbits=forbits)
        case "fges":
            forbits = call_graph.get_forbits(dataset, pk) if enable_prior_knowledge else []
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                graph = FGES(FGESConfig(run_pdag2dag=True)).train(dataset, forbits=forbits)
        case "ges":
            forbits = call_graph.get_forbits(dataset, pk) if enable_prior_knowledge else []
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                graph = GES(GESConfig(run_pdag2dag=True)).train(dataset, forbits=forbits)
        case "call_graph":
            nodes: mn.MetricNodes = mn.MetricNodes.from_dataframe(dataset)
            init_dg = call_graph.prepare_init_graph(nodes, pk, enable_prior_knowledge=enable_prior_knowledge, enable_orientation=True)
            init_dg = nx.relabel_nodes(init_dg, {n: n.label for n in init_dg.nodes}, copy=False)
            graph = nx.to_pandas_adjacency(init_dg)
        case _:
            raise ValueError(f"Unknown localization method: {method}")

    anomalous_metrics_for_walk: list[str] | None = []
    if anomalous_metrics is not None:
        for anomalous_metric in anomalous_metrics:
            if anomalous_metric in graph.columns:
                anomalous_metrics_for_walk.append(anomalous_metric)
    if len(anomalous_metrics_for_walk) == 0:
        anomalous_metrics_for_walk = None

    match walk_method:
        case "ht":
            model = HT(config=HTConfig(graph=graph, root_cause_top_k=root_cause_top_k))
            model.train(normal_df)
            results = model.find_root_causes(
                anomalous_df,
                anomalous_metrics_for_walk[0] if anomalous_metrics_for_walk is not None else None,
                adjustment=True,
            ).to_list()
        case "pagerank":
            rank = nx.pagerank(nx.DiGraph(graph).reverse())
            results = [{"root_cause": k, "score": v} for k, v in sorted(rank.items(), key=lambda item: item[1], reverse=True)][:root_cause_top_k]
        case "rw-2":
            if anomalous_metrics is None:
                return []
            model = RandomWalk(config=RandomWalkConfig(graph=graph, root_cause_top_k=root_cause_top_k, use_partial_corr=False))
            results = model.find_root_causes(anomalous_metrics_for_walk, anomalous_df).to_list()
        case _:
            raise ValueError(f"Unknown walk method: {walk_method}")

    return [(r["root_cause"], r["score"]) for r in results]
