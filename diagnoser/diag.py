import time
import warnings
from itertools import combinations
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from causality.inference.independence_tests import RobustRegressionTest
from causality.inference.search import IC
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc as cl_pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

warnings.filterwarnings("ignore", message=r"^No GPU automatically detected")

from causallearn.utils import cit
from cdt.causality.graph import GIES as cdt_GIES
from cdt.causality.graph import PC as cdt_PC
from cdt.causality.graph import LiNGAM
from sklearn.preprocessing import KBinsDiscretizer

# import diagnoser.causalgraph.causallearn_cit_fisherz_patch  # noqa: F401  for only patching
import diagnoser.causalgraph.cdt_PC_patch  # noqa: F401  for only patching
import diagnoser.metric_node as mn
from diagnoser import causalrca, nx_util, rcd
from diagnoser.causalgraph.pcalg import estimate_cpdag, estimate_skeleton
from diagnoser.causalgraph.pcalg_patch import estimate_skeleton_with_indep_test
from diagnoser.causalgraph.pgmpy_PC import PC
from diagnoser.citest.fisher_z import ci_test_fisher_z
from diagnoser.citest.fisher_z_pgmpy import fisher_z
from meltria.priorknowledge.priorknowledge import PriorKnowledge
from tsdr.clustering.pearsonr import pearsonr_left_shift

# from .citest.rlm import citest_rlm


def _discretize(data: pd.DataFrame, bins: int, standardize: bool = False):
    if standardize:
        data = data.apply(scipy.stats.zscore)
    discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans")
    discretizer.fit(data)
    disc_d = discretizer.transform(data)
    disc_d = pd.DataFrame(disc_d, columns=data.columns.values.tolist())
    for c in disc_d:
        disc_d[c] = disc_d[c].astype(int)
    return disc_d


def filter_by_target_metrics(data_df: pd.DataFrame, pk: PriorKnowledge) -> pd.DataFrame:
    """Filter by specified target metrics"""
    containers_df, services_df, nodes_df, middlewares_df = None, None, None, None
    target: dict[str, list[str]] = pk.get_diagnoser_target_data()

    if pk.target_metric_types["containers"]:
        containers_df = data_df.filter(
            regex=f"^c-.+({'|'.join(target['containers'])})$"
        )
    if pk.target_metric_types["services"]:
        services_df = data_df.filter(regex=f"^s-.+({'|'.join(target['services'])})$")
    if pk.target_metric_types["nodes"]:
        nodes_df = data_df.filter(regex=f"^n-.+({'|'.join(target['nodes'])})$")
    if pk.target_metric_types["middlewares"]:
        middlewares_df = data_df.filter(
            regex=f"^m-.+({'|'.join(target['middlewares'])})$"
        )

    if len(skip_ctnrs := pk.get_skip_containers()) > 0:
        if containers_df is not None:
            cols = containers_df.filter(
                regex=f"^c-({'|'.join(skip_ctnrs)})_.+"
            ).columns.tolist()
            containers_df.drop(columns=cols, inplace=True)
        if middlewares_df is not None:
            cols = middlewares_df.filter(
                regex=f"^m-({'|'.join(skip_ctnrs)})_.+"
            ).columns.tolist()
            middlewares_df.drop(columns=cols, inplace=True)
    if len(skip_services := pk.get_skip_services()) > 0:
        if services_df is not None:
            cols = services_df.filter(
                regex=f"^s-({'|'.join(skip_services)})_.+"
            ).columns.tolist()
            services_df.drop(columns=cols, inplace=True)

    return pd.concat(
        [
            df
            for df in [containers_df, services_df, nodes_df, middlewares_df]
            if df is not None
        ],
        axis=1,
    )


def build_subgraph_of_removal_edges(
    nodes: mn.MetricNodes, pk: PriorKnowledge
) -> nx.Graph:
    """Build a subgraph consisting of removal edges with prior knowledges."""
    ctnr_graph: nx.Graph = pk.get_container_call_digraph().to_undirected()
    service_graph: nx.Graph = pk.get_service_call_digraph().to_undirected()
    node_ctnr_graph: nx.Graph = pk.get_nodes_to_containers_graph()

    G: nx.Graph = nx.Graph()
    for u, v in combinations(nodes, 2):
        # "container" and "middleware" is the same.
        if (u.is_container() or u.is_middleware()) and (
            v.is_container() or v.is_middleware()
        ):
            if u.comp == v.comp or ctnr_graph.has_edge(u.comp, v.comp):
                continue
        elif (u.is_container() or u.is_middleware()) and v.is_service():
            u_service: str = pk.get_service_by_container(u.comp)
            if u_service == v.comp or service_graph.has_edge(u_service, v.comp):
                continue
        elif u.is_service() and (v.is_container() or v.is_middleware()):
            v_service: str = pk.get_service_by_container(v.comp)
            if u.comp == v_service or service_graph.has_edge(u.comp, v_service):
                continue
        elif u.is_service() and v.is_service():
            if u.comp == v.comp or service_graph.has_edge(u.comp, v.comp):
                continue
        elif u.is_node() and v.is_node():
            # each node has no connectivity.
            pass
        elif u.is_node() and (v.is_container() or v.is_middleware()):
            if node_ctnr_graph.has_edge(u.comp, v.comp):
                continue
        elif (u.is_container() or u.is_middleware()) and v.is_node():
            if node_ctnr_graph.has_edge(u.comp, v.comp):
                continue
        elif u.is_node() and v.is_service():
            v_ctnrs: list[str] = pk.get_service_containers(v.comp)
            has_ctnr_on_node = False
            for v_ctnr in v_ctnrs:
                if node_ctnr_graph.has_edge(u.comp, v_ctnr):
                    has_ctnr_on_node = True
                    break
            if has_ctnr_on_node:
                continue
        elif u.is_service() and v.is_node():
            u_ctnrs: list[str] = pk.get_service_containers(u.comp)
            has_ctnr_on_node = False
            for u_ctnr in u_ctnrs:
                if node_ctnr_graph.has_edge(u_ctnr, v.comp):
                    has_ctnr_on_node = True
                    break
            if has_ctnr_on_node:
                continue
        else:
            raise ValueError(f"'{u}' and '{v}' is an unexpected pair")
        # use node number because 'pgmpy' package handles only graph nodes consisted with numpy array.
        G.add_edge(u, v)
    return G


def prepare_init_graph(
    nodes: mn.MetricNodes,
    pk: PriorKnowledge,
    enable_orientation: bool = False,
    enable_prior_knowledge: bool = True,
) -> nx.Graph:
    """Prepare initialized causal graph."""
    init_g = nx.Graph()
    for u, v in combinations(nodes, 2):
        init_g.add_edge(u, v)
    if enable_prior_knowledge:
        RG: nx.Graph = build_subgraph_of_removal_edges(nodes, pk)
        init_g.remove_edges_from(RG.edges())
    if enable_orientation:
        return fix_edge_directions_in_causal_graph(init_g, pk)
    return init_g


def fix_edge_direction_based_hieralchy(
    G: nx.DiGraph, u: mn.MetricNode, v: mn.MetricNode, pk: PriorKnowledge
) -> None:
    # Force direction from (container -> service) to (service -> container) in same service
    if u.is_service() and v.is_container():
        # check whether u and v in the same service
        v_service = pk.get_service_by_container(v.comp)
        if u.comp == v_service:
            nx_util.reverse_edge_direction(G, u, v)


def fix_edge_direction_based_network_call(
    G: nx.DiGraph,
    u: mn.MetricNode,
    v: mn.MetricNode,
    service_dep_graph: nx.DiGraph,
    container_dep_graph: nx.DiGraph,
    pk: PriorKnowledge,
) -> None:
    # From service to service
    if u.is_service() and v.is_service():
        # If u and v is in the same service, force bi-directed edge.
        if u.comp == v.comp:
            nx_util.set_bidirected_edge(G, u, v)
        elif (v.comp not in service_dep_graph[u.comp]) and (
            u.comp in service_dep_graph[v.comp]
        ):
            nx_util.reverse_edge_direction(G, u, v)

    # From container to container
    if u.is_container_or_middleware() and v.is_container_or_middleware():
        # If u and v is in the same container, force bi-directed edge.
        if u.comp == v.comp:
            nx_util.set_bidirected_edge(G, u, v)
        elif (v.comp not in container_dep_graph[u.comp]) and (
            u.comp in container_dep_graph[v.comp]
        ):
            nx_util.reverse_edge_direction(G, u, v)

    # From service to container
    if u.is_service() and v.is_container_or_middleware():
        v_service = pk.get_service_by_container(v.comp)
        if (v_service not in service_dep_graph[u.comp]) and (
            u.comp in service_dep_graph[v_service]
        ):
            nx_util.reverse_edge_direction(G, u, v)

    # From container to service
    if u.is_container_or_middleware() and v.is_service():
        u_service = pk.get_service_by_container(u.comp)
        if (v.comp not in service_dep_graph[u_service]) and (
            u_service in service_dep_graph[v.comp]
        ):
            nx_util.reverse_edge_direction(G, u, v)


def fix_edge_directions_in_causal_graph(
    G: nx.Graph | nx.DiGraph,
    pk: PriorKnowledge,
) -> nx.DiGraph:
    """Fix the edge directions in the causal graphs.
    1. Fix directions based on the system hieralchy such as a service and a container
    2. Fix directions based on the network call graph.
    """
    if not G.is_directed():
        G = G.to_directed()
    service_dep_graph: nx.DiGraph = pk.get_service_call_digraph().reverse()
    container_dep_graph: nx.DiGraph = pk.get_container_call_digraph().reverse()
    # Traverse the all edges of G via the neighbors
    for u, nbrsdict in G.adjacency():
        nbrs = list(
            nbrsdict.keys()
        )  # to avoid 'RuntimeError: dictionary changed size during iteration'
        for v in nbrs:
            # u -> v
            fix_edge_direction_based_hieralchy(G, u, v, pk)
            fix_edge_direction_based_network_call(
                G, u, v, service_dep_graph, container_dep_graph, pk
            )
    return G


def build_causal_graph_with_pcalg(
    dm: np.ndarray,
    nodes: mn.MetricNodes,
    init_g: nx.Graph,
    pk: PriorKnowledge,
    pc_citest_alpha: float,
    pc_variant: str = "",
    pc_citest: str = "fisher-z",
    disable_orientation: bool = False,
    disable_ci_edge_cut: bool = False,
    use_indep_test_instead_of_ci: bool = False,
) -> nx.Graph:
    """
    Build causal graph with PC algorithm.
    """
    init_g = nx.relabel_nodes(init_g, mapping=nodes.node_to_num)
    cm = np.corrcoef(dm.T)
    ci_test: str | Callable
    match pc_citest:
        case "fisher-z":
            ci_test = ci_test_fisher_z
        case "rlm":
            ci_test = citest_rlm
        case _:
            ci_test = pc_citest
    if use_indep_test_instead_of_ci:
        G = estimate_skeleton_with_indep_test(
            indep_test_func=ci_test,
            data_matrix=dm,
            alpha=pc_citest_alpha,
            corr_matrix=cm,
            init_graph=init_g,
            disable_ci_edge_cut=disable_ci_edge_cut,
            use_indep_test_instead_of_ci=use_indep_test_instead_of_ci,
        )
    else:
        (G, sep_set) = estimate_skeleton(
            indep_test_func=ci_test,
            data_matrix=dm,
            alpha=pc_citest_alpha,
            corr_matrix=cm,
            init_graph=init_g,
            method=pc_variant,
            disable_ci_edge_cut=disable_ci_edge_cut,
            use_indep_test_instead_of_ci=use_indep_test_instead_of_ci,
        )
    if disable_orientation:
        return nx.relabel_nodes(G, mapping=nodes.num_to_node)
    else:
        if not use_indep_test_instead_of_ci:
            G = estimate_cpdag(skel_graph=G, sep_set=sep_set)
        G = nx.relabel_nodes(G, mapping=nodes.num_to_node)
        return fix_edge_directions_in_causal_graph(G, pk)


def build_causal_graphs_with_pgmpy(
    df: pd.DataFrame,
    pk: PriorKnowledge,
    pc_citest_alpha: float,
    pc_variant: str = "orig",
    pc_citest: str = "fisher-z",
    disable_orientation: bool = False,
) -> nx.Graph:
    c = PC(data=df)
    ci_test = fisher_z if pc_citest == "fisher-z" else pc_citest
    result = c.estimate(
        variant=pc_variant,
        ci_test=ci_test,
        significance_level=pc_citest_alpha,
        return_type="skeleton" if disable_orientation else "pdag",
    )
    G: nx.Graph
    if disable_orientation:
        G, _sep_set = result
    else:
        G = result
        G = fix_edge_directions_in_causal_graph(G, pk)
    return G


def build_causal_graphs_with_cdt(
    df: pd.DataFrame,
    init_g: nx.Graph,
    pk: PriorKnowledge,
    cg_algo: str = "pc",
    pc_citest_alpha: float = 0.05,
    pc_citest: str = "gaussian",
    pc_njobs: int | None = None,
    disable_orientation: bool = False,
) -> nx.Graph:
    match cg_algo:
        case "pc":
            pc = cdt_PC(CItest=pc_citest, alpha=pc_citest_alpha, njobs=pc_njobs)
            # create_graph_from_init_graph is a patched method.
            G = pc.create_graph_from_init_graph(df, init_graph=init_g)
        case "gies":
            gies = cdt_GIES(score="obs")
            G = gies.create_graph_from_init_graph(df, init_graph=init_g)
        case "lingam":
            lingam = LiNGAM()
            G = lingam.create_graph_from_data(df)
        case _:
            raise ValueError(f"Unsupported causal graph algorithm: {cg_algo}")
    G = mn.relabel_graph_labels_to_node(G)
    return G if disable_orientation else fix_edge_directions_in_causal_graph(G, pk)


def build_causal_graphs_with_causallearn(
    df: pd.DataFrame,
    nodes: mn.MetricNodes,
    init_g: nx.Graph,
    pk: PriorKnowledge,
    cg_algo: str = "pc",
    pc_citest_alpha: float = 0.05,
    pc_citest: str = "fisherz",
    pc_citest_bins: int = 5,
    pc_stable: bool = True,
    disable_orientation: bool = False,
) -> nx.Graph:
    init_dg = fix_edge_directions_in_causal_graph(init_g, pk)
    background_knowledge = BackgroundKnowledge()
    for node1, node2 in combinations(init_dg.nodes, 2):
        if not init_dg.has_edge(node1, node2):
            background_knowledge = background_knowledge.add_forbidden_by_node(
                GraphNode(node1.label), GraphNode(node2.label)
            )
        if not init_dg.has_edge(node2, node1):
            background_knowledge = background_knowledge.add_forbidden_by_node(
                GraphNode(node2.label), GraphNode(node1.label)
            )

    match pc_citest:
        case "fisherz":
            indep_test = cit.fisherz
        case "mv_fisherz":
            indep_test = cit.mv_fisherz
        case "chisq":
            df = _discretize(df, bins=pc_citest_bins, standardize=True)
            indep_test = cit.chisq
        case "gsq":
            df = _discretize(df, bins=pc_citest_bins, standardize=True)
            indep_test = cit.gsq
        case _:
            raise ValueError(f"Unsupported independence test: {pc_citest}")

    G: nx.Graph
    match cg_algo:
        case "pc":
            G_ = cl_pc(
                data=df.to_numpy(dtype=np.float32),
                alpha=pc_citest_alpha,
                indep_test=indep_test,
                stable=pc_stable,
                uc_rule=2,  # definiteMaxP
                uc_priority=2,
                mvpc=False,
                # node_names=df.columns.to_list(),
                background_knowledge=background_knowledge,
                verbose=False,
                show_progress=False,
            )
            G_.to_nx_graph()
            G = nx.relabel_nodes(
                G_.nx_graph, mapping=nodes.num_to_node_for_causallearn()
            )
        case "fci":
            G_, edges = fci(
                dataset=df.to_numpy(dtype=np.float32),
                alpha=pc_citest_alpha,
                indep_test=pc_citest,
                # node_names=df.columns.to_list(),
                background_knowledge=background_knowledge,
                verbose=False,
                show_progress=False,
            )
            G = nx.relabel_nodes(G_, mapping=nodes.num_to_node_for_causallearn())
        case _:
            raise ValueError(f"Unsupported causal graph algorithm: {cg_algo}")
    return G if disable_orientation else fix_edge_directions_in_causal_graph(G, pk)


def build_causal_graph_with_causality(
    df: pd.DataFrame,
    init_g: nx.Graph,
    pk: PriorKnowledge,
    pc_citest_alpha: float = 0.05,
    disable_orientation: bool = False,
) -> nx.Graph:
    variable_types = {col: "c" for col in df.columns}
    ic_algorithm = IC(RobustRegressionTest, alpha=pc_citest_alpha)
    G = ic_algorithm.search(df, variable_types)
    assert G is not None, "IC algorithm failed to find a causal graph"
    G = mn.relabel_graph_labels_to_node(G)
    return G if disable_orientation else fix_edge_directions_in_causal_graph(G, pk)


def fisher_z_two_var(dm, cm, x, y) -> float:
    m = dm.shape[0]
    r = cm[x, y]
    if 1 - r == 0:
        r = 1 - 1e-10
    if r >= 1.0:
        r = 1.0 - 1e-10
    elif r <= -1.0:
        r = -1.0 + 1e-10
    z = np.log((1 + r) / (1 - r))
    zstat = np.sqrt(m - 3) * 0.5 * np.abs(z)
    p_val = 2.0 * (1 - scipy.stats.norm.cdf(np.abs(zstat)))
    return p_val


def build_causal_graph_without_citest(
    data_df: pd.DataFrame,
    init_g: nx.Graph,
    pk: PriorKnowledge,
    pc_citest: str,
    pc_citest_alpha: float,
    disable_orientation: bool = False,
) -> nx.Graph:
    nodes = mn.MetricNodes.from_dataframe(data_df)
    dm = data_df.to_numpy(dtype=np.float32)
    cm = np.corrcoef(dm.T)
    citest_func = fisher_z_two_var

    G = nx.relabel_nodes(init_g, mapping=nodes.node_to_num)
    for u, v in G.edges:
        p_val = citest_func(dm, cm, u, v)
        if p_val > pc_citest_alpha:
            G.remove_edge(u, v)

    G = nx.relabel_nodes(G, mapping=nodes.num_to_node)
    return (
        G
        if disable_orientation
        else fix_edge_directions_in_causal_graph(G.to_directed(), pk)
    )


def find_connected_subgraphs(
    G: nx.Graph, root_labels: tuple[str, ...]
) -> tuple[list[nx.Graph], list[nx.Graph]]:
    """Find subgraphs connected components."""
    root_contained_subg: list[nx.Graph] = []
    root_uncontained_subg: list[nx.Graph] = []
    root_nodes = [mn.MetricNode(root) for root in root_labels]
    for c in nx.connected_components(G.to_undirected()):
        subg = G.subgraph(c).copy()
        if any([r in c for r in root_nodes]):
            root_contained_subg.append(subg)
        else:
            root_uncontained_subg.append(subg)
    return root_contained_subg, root_uncontained_subg


def remove_nodes_subgraph_uncontained_root(
    G: nx.Graph, root_labels: tuple[str, ...]
) -> nx.Graph:
    """Find graphs containing root metric node."""
    remove_nodes = []
    UG: nx.Graph = G.to_undirected()
    for node in G.nodes:
        has_paths: list[bool] = []
        for root in root_labels:
            rmn = mn.MetricNode(root)
            if UG.has_node(rmn) and UG.has_node(node):
                has_paths.append(nx.has_path(UG, rmn, node))
        if not any(has_paths):
            remove_nodes.append(node)
            continue
    G.remove_nodes_from(remove_nodes)
    return G


def build_causal_graph_with_library(
    dataset: pd.DataFrame,
    nodes: mn.MetricNodes,
    init_g: nx.Graph,
    pk: PriorKnowledge,
    pc_library: str,
    **kwargs: Any,
) -> nx.Graph:
    G: nx.Graph
    match (pc_library):
        case "pcalg":
            G = build_causal_graph_with_pcalg(
                dataset.to_numpy(),
                nodes,
                init_g,
                pk,
                pc_variant=kwargs["pc_variant"],
                pc_citest=kwargs["pc_citest"],
                pc_citest_alpha=kwargs["pc_citest_alpha"],
                disable_orientation=kwargs["disable_orientation"],
                disable_ci_edge_cut=kwargs["disable_ci_edge_cut"],
                use_indep_test_instead_of_ci=kwargs["use_indep_test_instead_of_ci"],
            )
        case "pgmpy":
            G = build_causal_graphs_with_pgmpy(
                dataset,
                pk,
                pc_variant=kwargs["pc_variant"],
                pc_citest=kwargs["pc_citest"],
                pc_citest_alpha=kwargs["pc_citest_alpha"],
                disable_orientation=kwargs["disable_orientation"],
            )
        case "cdt":
            G = build_causal_graphs_with_cdt(
                dataset,
                init_g,
                pk,
                cg_algo=kwargs["cg_algo"],
                pc_citest=kwargs["pc_citest"],
                pc_citest_alpha=kwargs["pc_citest_alpha"],
                pc_njobs=kwargs.get("pc_njobs"),
                disable_orientation=kwargs["disable_orientation"],
            )
        case "causallearn":
            G = build_causal_graphs_with_causallearn(
                dataset,
                nodes,
                init_g,
                pk,
                cg_algo=kwargs["cg_algo"],
                pc_citest=kwargs["pc_citest"],
                pc_citest_alpha=kwargs["pc_citest_alpha"],
                pc_citest_bins=kwargs["pc_citest_bins"],
                pc_stable=kwargs["pc_stable"],
                disable_orientation=kwargs["disable_orientation"],
            )
        case "causality":
            G = build_causal_graph_with_causality(
                dataset,
                init_g=init_g,
                pk=pk,
                pc_citest_alpha=kwargs["pc_citest_alpha"],
                disable_orientation=kwargs["disable_orientation"],
            )
        case "without_citest":
            G = build_causal_graph_without_citest(
                dataset,
                init_g,
                pk,
                pc_citest=kwargs["pc_citest"],
                pc_citest_alpha=kwargs["pc_citest_alpha"],
                disable_orientation=kwargs["disable_orientation"],
            )
        case _:
            raise ValueError(f"pc_library should be pcalg or pgmpy ({pc_library})")
    return G


def build_causal_graph(
    dataset: pd.DataFrame,
    pk: PriorKnowledge,
    enable_prior_knowledge: bool = True,
    use_call_graph: bool = False,
    use_complete_graph: bool = False,
    **kwargs: Any,
) -> tuple[nx.Graph, tuple[list[nx.Graph], list[nx.Graph]], dict[str, Any]]:
    assert not (
        use_call_graph and use_complete_graph
    ), "use_call_graph and use_complete_graph cannot be True at the same time"

    if use_call_graph:
        assert (
            enable_prior_knowledge
        ), "use_call_graph=True requires enable_prior_knowledge=True"
    if use_complete_graph:
        assert (
            not enable_prior_knowledge
        ), "use_complete_graph=True requires enable_prior_knowledge=False"

    dataset = filter_by_target_metrics(dataset, pk)
    if not any(label in dataset.columns for label in pk.get_root_metrics()):
        raise ValueError(f"dataset has no root metric node: {pk.get_root_metrics()}")

    building_graph_start: float = time.time()

    nodes: mn.MetricNodes = mn.MetricNodes.from_dataframe(dataset)
    init_g: nx.Graph = prepare_init_graph(
        nodes, pk, enable_prior_knowledge=enable_prior_knowledge
    )

    if use_call_graph or use_complete_graph:
        G = fix_edge_directions_in_causal_graph(init_g, pk)
    else:
        G = build_causal_graph_with_library(
            dataset,
            nodes,
            init_g,
            pk,
            **kwargs,
        )

    root_contained_graphs, root_uncontained_graphs = find_connected_subgraphs(
        G, pk.get_root_metrics()
    )

    building_graph_elapsed: float = time.time() - building_graph_start

    G = remove_nodes_subgraph_uncontained_root(G, pk.get_root_metrics())  # for stats
    if G.size() == 0:
        return G, (root_contained_graphs, root_uncontained_graphs), {}
    stats = {
        "init_graph_nodes_num": init_g.number_of_nodes(),
        "init_graph_edges_num": init_g.number_of_edges(),
        "causal_graph_nodes_num": G.number_of_nodes(),
        "causal_graph_edges_num": G.number_of_edges(),
        "causal_graph_density": nx.density(G),
        "causal_graph_flow_hierarchy": nx.flow_hierarchy(G)
        if G.is_directed()
        else np.nan,
        "building_graph_elapsed_sec": building_graph_elapsed,
    }
    return G, (root_contained_graphs, root_uncontained_graphs), stats


def max_xcorr_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_dev = a - a.mean()
    b_dev = b - b.mean()
    xc_ab = np.correlate(a_dev, b_dev, mode="full")
    xc_ab /= np.linalg.norm(a_dev, ord=2) * np.linalg.norm(b_dev, ord=2)  # normalize
    return np.max(np.abs(xc_ab))


def xcorr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # expected X is 2darray, y is 1darray
    assert X.ndim == 2, f"X.ndim must be 2, but {X.ndim}"
    assert y.ndim == 1, f"y.ndim must be 1, but {y.ndim}"
    return np.apply_along_axis(lambda x: max_xcorr_pair(x, y), axis=1, arr=X)


def corr_pearsonr_left_shift(X: np.ndarray, y: np.ndarray, l_p: int) -> np.ndarray:
    # expected X is 2darray, y is 1darray
    assert X.ndim == 2, f"X.ndim must be 2, but {X.ndim}"
    assert y.ndim == 1, f"y.ndim must be 1, but {y.ndim}"
    return np.apply_along_axis(
        lambda x: pearsonr_left_shift(x, y, l_p=l_p, apply_abs=True), axis=1, arr=X
    )


def prepare_monitor_rank_based_random_walk(
    G: nx.DiGraph,
    dataset: pd.DataFrame,
    pk: PriorKnowledge,
    root_metric_type: str,
    corr_method: str = "pearsonr",  # or "xcorr"
    corr_left_shift_lp: int = 0,
) -> tuple[nx.DiGraph, dict[str, float]]:
    """MonitorRank-based ranked algorithm
    G must be a call graph, not causal graph
    """
    assert (
        G.number_of_nodes() > 1
    ), f"number of nodes must be > 1, but {G.number_of_nodes()}."

    G = mn.relabel_graph_nodes_to_label(G)
    data = dataset.filter(list(G.nodes), axis=1)
    front_root_metrics = [
        m for m in data.columns.tolist() if m in pk.get_root_metrics()
    ]
    assert (
        len(front_root_metrics) > 0
    ), f"dataset has no root metric node: {pk.get_root_metrics()}"
    start_front_root_metric = pk.get_root_metric_by_type(root_metric_type)
    if start_front_root_metric not in front_root_metrics:
        start_front_root_metric = front_root_metrics[0]
    start_front_root_metric_id = data.columns.tolist().index(start_front_root_metric)
    assert (
        start_front_root_metric_id >= 0
    ), f"dataset has no root metric node: {start_front_root_metric}"

    match corr_method:
        case "pearsonr":
            corr = np.corrcoef(data.values.T)  # calculate pearson correlation
            corr = corr[start_front_root_metric_id]
        case "xcorr":
            corr = np.corrcoef(data.values.T)  # calculate pearson correlation
            corr = xcorr(
                data.values.T, data.values[:, start_front_root_metric_id]
            )  # calculate cross correlation
        case "left_shift":
            corr = corr_pearsonr_left_shift(
                X=data.values.T,
                y=data.values[:, start_front_root_metric_id],
                l_p=corr_left_shift_lp,
            )
        case _:
            raise ValueError(
                f"corr_method must be 'pearsonr' or 'xcorr', or 'left_shift', but {corr_method}"
            )

    sim = [abs(x) for x in corr]  # similarity to front root metric
    rho = 0.1
    # 'weight' of each edge means "transition probability"
    for i in G.nodes:
        for j in G.nodes:
            s_i = sim[list(G.nodes).index(i)]
            s_j = sim[list(G.nodes).index(j)]
            if G.has_edge(i, j):  # forward edge
                G.edges[i, j]["weight"] = abs(s_j)
            elif G.has_edge(j, i):  # backward edge
                G.add_edge(i, j, weight=rho * abs(s_i))

    # self edge
    for i in G.nodes:
        if i in front_root_metrics:
            continue
        s_i = sim[list(G.nodes).index(i)]
        p_i: list[float] = [G[i][j]["weight"] for j in G[i]]
        G.add_edge(i, i, weight=max(0, s_i - max(p_i)))

    # normalize
    for i in G.nodes:
        adj_sum = sum([G[i][j]["weight"] for j in G[i]])
        for j in G[i]:
            G.edges[i, j]["weight"] /= adj_sum

    u = {
        n: sim[list(G.nodes).index(n)] for n in G.nodes if n != start_front_root_metric
    }  # preference vector
    u[start_front_root_metric] = 0

    return G, u


def walk_causal_graph_with_monitorrank(
    G: nx.DiGraph,
    dataset: pd.DataFrame,
    pk: PriorKnowledge,
    root_metric_type: str,
    **kwargs: Any,
) -> tuple[nx.DiGraph, list[tuple[str, float]]]:
    modified_g, preference_vector = prepare_monitor_rank_based_random_walk(
        G.reverse(),
        dataset,
        pk,
        root_metric_type,
        corr_method=kwargs["corr_method"],  # type: ignore
        corr_left_shift_lp=kwargs["corr_left_shift_lp"],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        pr = nx.pagerank(
            modified_g,
            alpha=kwargs.get("pagerank_alpha", 0.85),  # type: ignore
            weight="weight",
            personalization=preference_vector,
        )
    # sort by rank
    return modified_g, sorted(pr.items(), key=lambda item: item[1], reverse=True)


def build_and_walk_causal_graph(
    dataset: pd.DataFrame,
    pk: PriorKnowledge,
    root_metric_type: str,  # "latency" or "throughput" or "error"
    enable_prior_knowledge: bool = True,
    use_call_graph: bool = False,
    use_complete_graph: bool = False,
    use_causalrca: bool = False,
    use_rcd: bool = False,
    **kwargs: Any,
) -> tuple[nx.Graph, list[tuple[str, float]]]:
    assert not (
        use_call_graph and use_complete_graph
    ), "use_call_graph and use_complete_graph cannot be True at the same time."

    assert not (
        use_causalrca and use_rcd
    ), "use_causalrca and use_rcd cannot be True at the same time."
    if use_causalrca:
        return causalrca.build_and_walk_causal_graph(dataset, **kwargs)
    if use_rcd:
        return nx.empty_graph(n=0), rcd.localize(dataset, **kwargs)

    G, (root_contained_graphs, root_uncontained_graphs), stats = build_causal_graph(
        dataset,
        pk,
        enable_prior_knowledge=enable_prior_knowledge,
        use_call_graph=use_call_graph,
        use_complete_graph=use_complete_graph,
        **kwargs,
    )
    max_graph = max(root_contained_graphs, key=lambda g: g.number_of_nodes())
    if max_graph.number_of_nodes() < 2:
        return max_graph, []

    target_graph = next(
        filter(
            lambda g: g.has_node(pk.get_root_metric_by_type(root_metric_type)),
            root_contained_graphs,
        ),
        None,
    )
    if target_graph is None:
        target_graph = max_graph
    elif target_graph.number_of_nodes() < 2:  # fallback to other root metrics
        target_graph = max_graph
    ranks: list[tuple[str, float]]
    match (walk_method := kwargs["walk_method"]):
        case "monitorrank":
            modified_g, ranks = walk_causal_graph_with_monitorrank(
                target_graph, dataset, pk, root_metric_type, **kwargs
            )
        case _:
            raise ValueError(f"walk_method should be monitorrank ({walk_method})")
    return modified_g, ranks
