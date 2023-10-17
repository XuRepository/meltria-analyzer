import itertools
import logging
import pathlib
from typing import Generator

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from pyrca.simulation import data_gen

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "synthetic"


def generate_synthetic_data(
    num_node: int,
    num_edge: int,
    num_normal_samples: int,
    num_abnormal_samples: int,
    anomaly_type: int,
    func_type: str = 'identity',
    noise_type: str = 'uniform',
    weight_generator: str = 'uniform',
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame, set[str]] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    graph_matrix = data_gen.DAGGen(
        data_gen.DAGGenConfig(num_node=num_node, num_edge=num_edge)
    ).gen()
    G = nx.DiGraph(graph_matrix)

    # transform node names from 0 to N-1 to X1 to XN
    no_of_var = graph_matrix.shape[0]
    original_names = [i for i in range(no_of_var)]
    node_names = [("X%d" % (i + 1)) for i in range(no_of_var)]
    mapping = dict(zip(original_names, node_names))
    G = nx.relabel_nodes(G, mapping)
    adjacency_df = pd.DataFrame({node_names[i]: graph_matrix[:, i] for i in range(len(node_names))}, index=node_names)

    # _VALID_NOISE = ["normal", "exponential", "uniform", "laplace"]
    # _VALID_FUNC = ["identity", "square", "sin", "tanh"]
    # _VALID_WEIGHT = ["normal", "uniform"]
    normal_data, parent_weights, noise_weights, func_form, noise_form = data_gen.DataGen(
        data_gen.DataGenConfig(
            dag=graph_matrix,
            func_type=func_type,
            noise_type=noise_type,
            weight_generator=weight_generator,
            num_samples=num_normal_samples,
        )
    ).gen()

    # --- Abnormal data ---config = data_gen.AnomalyDataGenConfig(
    _SLI = 0
    tau = 3
    baseline = normal_data[:, _SLI].mean()
    sli_sigma = normal_data[:, _SLI].std()
    threshold = tau * sli_sigma
    anomaly_data, fault = data_gen.AnomalyDataGen(
        data_gen.AnomalyDataGenConfig(
            parent_weights=parent_weights,
            noise_weights=noise_weights,
            func_type=func_form,
            noise_type=noise_form,
            threshold=threshold,
            baseline=baseline,
            anomaly_type=anomaly_type,
            num_samples=num_abnormal_samples,
        )
    ).gen()
    true_root_causes = [mapping[i] for i in np.where(fault != 0)[0]]
    anomaly_propagated_nodes = set()
    for root_cause in true_root_causes:
        paths = nx.all_simple_paths(G, source=root_cause, target=["X1"])
        for path in paths:
            for node in path:
                anomaly_propagated_nodes.add(node)

    normal_data_df = pd.DataFrame(normal_data, columns=node_names)
    abnormal_data_df = pd.DataFrame(anomaly_data, columns=node_names)

    return normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes


def generate_datasets_from_param_sets(
    anomaly_types: list[int],
    data_scale_params: list[dict[str, int]],
    func_types: list[str],
    noise_types: list[str],
    weight_generators: list[str],
    num_trials: int,
    return_as: str = "list",
) -> list | Generator:
    params = list(itertools.product(
        anomaly_types, data_scale_params, func_types, noise_types, weight_generators, range(1, num_trials + 1),
    ))
    match return_as:
        case "list":
            datasets = joblib.Parallel(n_jobs=-1)(joblib.delayed(generate_synthetic_data)(
                num_node=data_scale_param["num_node"],
                num_edge=data_scale_param["num_edge"],
                num_normal_samples=data_scale_param["num_normal_samples"],
                num_abnormal_samples=data_scale_param["num_abnormal_samples"],
                anomaly_type=anomaly_type,
                func_type=func_type,
                noise_type=noise_type,
                weight_generator=weight_generator,
            ) for anomaly_type, data_scale_param, func_type, noise_type, weight_generator, _ in params)
            assert datasets is not None
            return list(zip(params, datasets))
        case "generator":
            for anomaly_type, data_scale_param, func_type, noise_type, weight_generator, trial_no in params:
                yield (anomaly_type, data_scale_param, func_type, noise_type, weight_generator, trial_no), generate_synthetic_data(
                    num_node=data_scale_param["num_node"],
                    num_edge=data_scale_param["num_edge"],
                    num_normal_samples=data_scale_param["num_normal_samples"],
                    num_abnormal_samples=data_scale_param["num_abnormal_samples"],
                    anomaly_type=anomaly_type,
                    func_type=func_type,
                    noise_type=noise_type,
                    weight_generator=weight_generator,
                )
            return
        case _:
            raise ValueError(f"Invalid return_as: {return_as}")


def sweep_and_save_generated_data(
    anomaly_types: list[int],
    data_scale_params: list[dict[str, int]],
    func_types: list[str],
    noise_types: list[str],
    weight_generators: list[str],
    trial_nos: list[int],
    n_jobs: int = -1,
):
    def gen(anomaly_type, data_params, func_type, noise_type, weight_generator, trial_no):
        logger.info(f"Running anomaly type {anomaly_type} with data scale {data_params}, func_type: {func_type}, noise_type: {noise_type}, weight_generator:{weight_generator}, trian_no {trial_no}...")

        normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes = generate_synthetic_data(
            num_node=data_params["num_node"],
            num_edge=data_params["num_edge"],
            num_normal_samples=data_params["num_normal_samples"],
            num_abnormal_samples=data_params["num_abnormal_samples"],
            anomaly_type=anomaly_type,
            func_type=func_type,
            noise_type=noise_type,
            weight_generator=weight_generator,
        )
        # save
        data_id = joblib.hash([anomaly_type, data_params, func_type, noise_type, weight_generator, trial_no], hash_name="sha1")
        assert data_id is not None
        (DATA_DIR / data_id).mkdir(parents=True, exist_ok=True)

        for obj, name in (
            (normal_data_df, "normal_data_df"),
            (abnormal_data_df, "abnormal_data_df"),
            (true_root_causes, "true_root_causes"),
            (adjacency_df, "adjacency_df"),
            (anomaly_propagated_nodes, "anomaly_propagated_nodes"),
        ):
            joblib.dump(obj, DATA_DIR / data_id / f"{name}.pkl.bz2", compress=("bz2", 3))

    params = list(itertools.product(
        anomaly_types, data_scale_params, func_types, noise_types, weight_generators, trial_nos,
    ))
    joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
        joblib.delayed(gen)(*param) for param in params
    )


def load_data(anomaly_type, data_params, func_type, noise_type, weight_generator, trial_no):
    data_id = joblib.hash([anomaly_type, data_params, func_type, noise_type, weight_generator, trial_no], hash_name="sha1")
    assert data_id is not None
    normal_data_df = joblib.load(DATA_DIR / data_id / "normal_data_df.pkl.bz2")
    abnormal_data_df = joblib.load(DATA_DIR / data_id / "abnormal_data_df.pkl.bz2")
    true_root_causes = joblib.load(DATA_DIR / data_id / "true_root_causes.pkl.bz2")
    adjacency_df = joblib.load(DATA_DIR / data_id / "adjacency_df.pkl.bz2")
    anomaly_propagated_nodes = joblib.load(DATA_DIR / data_id / "anomaly_propagated_nodes.pkl.bz2")
    return normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes
