import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.append((Path(__file__).parent.parent / "PyRCA").as_posix())

from pyrca.analyzers.epsilon_diagnosis import (EpsilonDiagnosis,
                                               EpsilonDiagnosisConfig)
from pyrca.analyzers.ht import HT, HTConfig
from pyrca.analyzers.random_walk import RandomWalk, RandomWalkConfig
from pyrca.analyzers.rcd import RCD, RCDConfig
from pyrca.graphs.causal.fges import FGES, FGESConfig
from pyrca.graphs.causal.lingam import LiNGAM, LiNGAMConfig
from pyrca.graphs.causal.pc import PC, PCConfig


def run_localization(
    dataset: pd.DataFrame,
    method: str,
    walk_method: str | None = None,
    root_cause_top_k: int = 15,
    anomalous_metrics: str | None = None,
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
            graph = PC(PCConfig(run_pdag2dag=True)).train(dataset)
        case "lingam":
            graph = LiNGAM(LiNGAMConfig(run_pdag2dag=True)).train(dataset)
        case "fges":
            graph = FGES(FGESConfig(run_pdag2dag=True)).train(dataset)
        case _:
            raise ValueError(f"Unknown localization method: {method}")

    match walk_method:
        case "ht":
            model = HT(config=HTConfig(graph=graph, root_cause_top_k=root_cause_top_k))
            model.train(normal_df)
            results = model.find_root_causes(anomalous_df, anomalous_metrics, adjustment=True).to_list()
        case "pagerank":
            rank = nx.pagerank(nx.DiGraph(graph).reverse())
            results = [{"root_cause": k, "score": v} for k, v in sorted(rank.items(), key=lambda item: item[1], reverse=True)][:root_cause_top_k]
        case "rw-2":
            model = RandomWalk(config=RandomWalkConfig(graph=graph, root_cause_top_k=root_cause_top_k, use_partial_corr=False))
            results = model.find_root_causes([anomalous_metrics], anomalous_df).to_list()
        case _:
            raise ValueError(f"Unknown walk method: {walk_method}")

    return [(r["root_cause"], r["score"]) for i, r in enumerate(results)]
