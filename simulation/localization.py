import gc
import itertools
import logging
import time
import warnings
from collections import defaultdict
from typing import Any, Final

import joblib
import networkx as nx
import pandas as pd
from pyrca.analyzers.bayesian import BayesianNetwork, BayesianNetworkConfig
from pyrca.analyzers.epsilon_diagnosis import (EpsilonDiagnosis,
                                               EpsilonDiagnosisConfig)
from pyrca.analyzers.ht import HT, HTConfig
from pyrca.analyzers.random_walk import RandomWalk, RandomWalkConfig
from pyrca.analyzers.rcd import RCD, RCDConfig
from pyrca.graphs.causal.ges import GES, GESConfig
from pyrca.graphs.causal.lingam import LiNGAM, LiNGAMConfig
from pyrca.graphs.causal.pc import PC, PCConfig
from pyrca.thirdparty.causallearn.utils.cit import gsq
from threadpoolctl import threadpool_limits

from simulation import feature_reduction
from simulation.synthetic_data import load_data

num_trials: int = 5
top_k: int = 10
LOCALIZATUON_METHODS: Final[list[str]] = [
    "RCD",
    "EpsilonDiagnosis",
    "PC+HT",
    "LiNGAM+HT",
    # "GES+HT",
    "PC+PageRank",
    "LiNGAM+PageRank",
    # "GES+PageRank",
    # "PC+RW-2",
    # "LiNGAM+RW-2",
    # "GES+RW-2",
]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)


def method_to_method_pair(method: str) -> tuple[str, str]:
    match method:
        case "RCD":
            return "rcd", ""
        case "EpsilonDiagnosis":
            return "epsilon_diagnosis", ""
        case "PC+HT":
            return "pc", "ht"
        case "LiNGAM+HT":
            return "lingam", "ht"
        case "GES+HT":
            return "ges", "ht"
        case "PC+PageRank":
            return "pc", "pagerank"
        case "LiNGAM+PageRank":
            return "lingam", "pagerank"
        case "GES+PageRank":
            return "ges", "pagerank"
        case "PC+RW-2":
            return "pc", "rw-2"
        case "LiNGAM+RW-2":
            return "lingam", "rw-2"
        case "GES+RW-2":
            return "ges", "rw-2"
        case _:
            raise ValueError(f"Unknown localization method: {method}")


def pr_k(
    results: dict[int, list[tuple[str, float]]],
    true_root_causes: dict[int, list[str]],
    top_k: int,
) -> dict[int, float]:
    k_values = range(1, top_k + 1)
    pr_at_k = {k: .0 for k in k_values}

    for fault_id, ranks in results.items():
        root_causes = true_root_causes[fault_id]
        for k in k_values:
            num_correct = sum([1 for metric, _ in ranks[:k] if metric in root_causes])
            pr_at_k[k] += (num_correct / min(k, len(root_causes)))

    for k in k_values:
        pr_at_k[k] /= len(results)

    return pr_at_k


def avg_k(
    results: dict[int, list[tuple[str, float]]],
    true_root_causes: dict[int, list[str]],
    top_k: int,
) -> dict[int, float]:
    k_values = range(1, top_k + 1)
    avg_at_k = {}
    for k in k_values:
        pr_at_k = pr_k(results, true_root_causes, top_k)
        avg_at_k[k] = sum([pr_at_k[j] for j in range(1, k + 1)]) / k
    return avg_at_k


def run_rcd(normal_data_df: pd.DataFrame, abnormal_data_df: pd.DataFrame, top_k: int, n_iters: int) -> list[tuple[str, float]]:
    model = RCD(config=RCDConfig(k=top_k, localized=True, ci_test=gsq))

    def _run_rcd() -> list[dict]:
        with threadpool_limits(limits=1):
            with warnings.catch_warnings(action='ignore', category=FutureWarning):
                return model.find_root_causes(normal_data_df, abnormal_data_df).to_list()

    if n_iters <= 1:
        return [(r["root_cause"], r["score"]) for r in _run_rcd()]
    # seed ensamble
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(_run_rcd)() for _ in range(n_iters))
    assert results is not None, "The results of rcd.rca_with_rcd are not empty"

    scores: dict[str, int] = defaultdict(int)
    for result in results:
        if result is None:
            continue
        for m in result[:top_k]:
            scores[m["root_cause"]] += 1
    return sorted([(metric, n / n_iters) for (metric, n) in scores.items()], key=lambda x: x[1], reverse=True)


def run_rca(
    data_df: pd.DataFrame,
    graph: pd.DataFrame,
    method: str,
    walk_method: str,
    top_k: int,
    boundary_index: int,
    abnormal_metrics: list[str] = ["X1"],
    **kwargs,
) -> list[tuple[str, float]]:
    if len(data_df.columns) == 0:
        return []
    normal_data_df = data_df[data_df.index < boundary_index]
    abnormal_data_df = data_df[data_df.index >= boundary_index]
    match method:
        case "epsilon_diagnosis":
            normal_df = normal_data_df.iloc[-len(abnormal_data_df):, :]
            model = EpsilonDiagnosis(config=EpsilonDiagnosisConfig(root_cause_top_k=top_k))
            with threadpool_limits(limits=1):
                model.train(normal_df)
                results = model.find_root_causes(abnormal_data_df).to_list()
            return [(r["root_cause"], r["score"]) for i, r in enumerate(results)]
        case "rcd":
            return run_rcd(normal_data_df, abnormal_data_df, top_k, kwargs.get("rcd_n_iters", 1))
        # case "BayesianNetwork":
        #     if "X1" not in normal_data_df.columns:
        #         return hits
        #     # estimated_matrix = PC(PCConfig(run_pdag2dag=True)).train(normal_data_df)
        #     model = BayesianNetwork(config=BayesianNetworkConfig(graph=graph, root_cause_top_k=top_k))
        #     model.train([normal_data_df, abnormal_data_df])
        #     # results = model.find_root_causes(anomalous_metrics=model.bayesian_model.nodes).to_list()
        #     results = model.find_root_causes(anomalous_metrics=["X1"]).to_list()
        case "pc":
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                with threadpool_limits(limits=1):
                    graph = PC(PCConfig(run_pdag2dag=True)).train(
                        pd.concat([normal_data_df, abnormal_data_df], axis=0, ignore_index=True),
                    )
        case "lingam":
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                with threadpool_limits(limits=1):
                    graph = LiNGAM(LiNGAMConfig(run_pdag2dag=True)).train(
                        pd.concat([normal_data_df, abnormal_data_df], axis=0, ignore_index=True),
                    )
        case "ges":
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                with threadpool_limits(limits=1):
                    graph = GES(GESConfig(run_pdag2dag=True)).train(
                        pd.concat([normal_data_df, abnormal_data_df], axis=0, ignore_index=True),
                    )
        case _:
            raise ValueError(f"Model {method} is not supported.")

    abnormal_metrics_for_walk: list[str] | None = []
    for abnormal_metric in abnormal_metrics:
        if abnormal_metric in graph.columns:
            abnormal_metrics_for_walk.append(abnormal_metric)
    if not abnormal_metrics_for_walk:
        abnormal_metrics_for_walk = None

    match walk_method:
        case "ht":
            model = HT(config=HTConfig(graph=graph, root_cause_top_k=top_k))
            with threadpool_limits(limits=1):
                model.train(normal_data_df)
                try:
                    results = model.find_root_causes(
                        abnormal_data_df,
                        abnormal_metrics_for_walk[0] if abnormal_metrics_for_walk is not None else None,
                        adjustment=True,
                    ).to_list()
                except nx.exception.NetworkXUnfeasible:
                    logger.warning("skip to run 'ht' because the graph has a cycle")
                    return []
        case "pagerank":
            with threadpool_limits(limits=1):
                rank = nx.pagerank(nx.DiGraph(graph).reverse())
                results = [{"root_cause": k, "score": v} for k, v in sorted(rank.items(), key=lambda item: item[1], reverse=True)][:top_k]
        case "rw-2":
            if abnormal_data_df is None or not abnormal_metrics_for_walk:
                return []
            model = RandomWalk(config=RandomWalkConfig(graph=graph, root_cause_top_k=top_k, use_partial_corr=False))
            with threadpool_limits(limits=1):
                results = model.find_root_causes(abnormal_metrics_for_walk, abnormal_data_df).to_list()
        case _:
            raise ValueError(f"Unknown walk method: {walk_method}")

    return [(r["root_cause"], r["score"]) for r in results]


def run_rca_with_params(
    method: str, top_k: int, trial_no: int, anomaly_type: int,
    data_scale_param: dict[str, int], func_type: str, noise_type: str, weight_generator: str,
    data_df, true_root_causes, graph,
    reduction_stats,
    **kwargs,
) -> list[dict[str, int]]:
    logger.info(f"Running {method} on anomaly type {anomaly_type} with data scale {data_scale_param}, func_type: {func_type}, noise_type: {noise_type}, weight_generator:{weight_generator}, trial_no {trial_no}...")

    causal_method, walk_method = method_to_method_pair(method)

    sta: float = time.perf_counter()

    ranks = run_rca(
        data_df, graph,
        method=causal_method, walk_method=walk_method,
        top_k=top_k, boundary_index=data_scale_param["num_normal_samples"],
        **kwargs,
    )

    end: float = time.perf_counter()
    elapsed: float = end - sta

    items = []
    for k, (metric, score) in enumerate(ranks, 1):
        hit = metric in true_root_causes
        items.append(
            dict(
                {
                    "trial_no": trial_no, "k": k, "metric": metric, "score": score,
                    "hit": hit,
                    "num_root_causes": len(true_root_causes),
                    "localization_method": method,
                    "elapsed_time_loc": elapsed,
                },
                **data_scale_param,
                **{"anomaly_type": anomaly_type, "func_type": func_type, "noise_type": noise_type, "weight_generator": weight_generator},
                **reduction_stats,
            )
        )
    return items


def sweep_reduction_and_localization(
    anomaly_types: list[int],
    data_scale_params: list[dict[str, int]],
    func_types: list[str],
    noise_types: list[str],
    weight_generators: list[str],
    trial_nos: list[int],
    localization_methods: list[str] = LOCALIZATUON_METHODS,
    n_jobs: int = -1,
    **localization_kwargs: dict[str, Any],
) -> pd.DataFrame:
    def _load_and_reduce_and_localize(anomaly_type, data_params, func_type, noise_type, weight_generator, trial_no):
        logger.info(f"Running anomaly type {anomaly_type} with data scale {data_params}, func_type: {func_type}, noise_type: {noise_type}, weight_generator:{weight_generator}, trian_no {trial_no}...")

        normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes = load_data(
            anomaly_type=anomaly_type,
            data_params=dict(
                num_node=data_params["num_node"],
                num_edge=data_params["num_edge"],
                num_normal_samples=data_params["num_normal_samples"],
                num_abnormal_samples=data_params["num_abnormal_samples"],
            ),
            func_type=func_type,
            noise_type=noise_type,
            weight_generator=weight_generator,
            trial_no=trial_no,
        )

        loc_results = []
        for fl_method in feature_reduction.REDUCTION_METHODS:
            logger.info(f"Running feature reduction method {fl_method} on anomaly type {anomaly_type} with data scale {data_params}, func_type: {func_type}, noise_type: {noise_type}, weight_generator:{weight_generator}, trian_no {trial_no}...")
            copied_normal_data_df = normal_data_df.copy()
            copied_abnormal_data_df = abnormal_data_df.copy()
            copied_adjacency_df = adjacency_df.copy()
            reduced_normal_df, reduced_abnormal_df, graph, stat = feature_reduction.reduce_features(
                fl_method, copied_normal_data_df, copied_abnormal_data_df,
                true_root_causes, copied_adjacency_df, anomaly_propagated_nodes,
            )
            del graph
            gc.collect()

            reduced_df = pd.concat([reduced_normal_df, reduced_abnormal_df], axis=0, ignore_index=True, copy=False)

            for loc_method in localization_methods:
                results = run_rca_with_params(
                    loc_method, top_k=top_k, trial_no=trial_no, anomaly_type=anomaly_type,
                    data_scale_param=data_params, func_type=func_type, noise_type=noise_type, weight_generator=weight_generator,
                    data_df=reduced_df, true_root_causes=true_root_causes, graph=copied_adjacency_df,
                    reduction_stats=stat,
                    **localization_kwargs,
                )
                loc_results.extend(results)

            del reduced_df, copied_normal_data_df, copied_abnormal_data_df, copied_adjacency_df
            gc.collect()

        del normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes
        gc.collect()

        return loc_results

    params = list(itertools.product(
        anomaly_types, data_scale_params, func_types, noise_types, weight_generators, trial_nos,
    ))
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_load_and_reduce_and_localize)(*param) for param in params
    )
    assert results is not None
    return pd.DataFrame(sum(results, []))
