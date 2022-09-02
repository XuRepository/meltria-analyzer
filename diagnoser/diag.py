import time
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pcalg
from pgmpy import estimators

import diagnoser.metric_node as mn
from diagnoser import nx_util
from meltria.priorknowledge.priorknowledge import PriorKnowledge

from .citest.fisher_z import ci_test_fisher_z
from .citest.fisher_z_pgmpy import fisher_z

SIGNIFICANCE_LEVEL = 0.05


def filter_by_target_metrics(data_df: pd.DataFrame, pk: PriorKnowledge) -> pd.DataFrame:
    """Filter by specified target metrics
    """
    containers_df, services_df, nodes_df, middlewares_df = None, None, None, None
    target: dict[str, list[str]] = pk.get_diagnoser_target_data()
    if 'containers' in target:
        containers_df = data_df.filter(
            regex=f"^c-.+({'|'.join(target['containers'])})$")
    if 'services' in target:
        services_df = data_df.filter(
            regex=f"^s-.+({'|'.join(target['services'])})$")
    if 'nodes' in target:
        nodes_df = data_df.filter(
            regex=f"^n-.+({'|'.join(target['nodes'])})$")
    if 'middlewares' in target:
        # TODO: middleware
        middlewares_df = data_df.filter(
            regex=f"^m-.+({'|'.join(target['middlewares'])})$")
    return pd.concat([containers_df, services_df, nodes_df], axis=1)


def build_subgraph_of_removal_edges(nodes: mn.MetricNodes, pk: PriorKnowledge) -> nx.Graph:
    """Build a subgraph consisting of removal edges with prior knowledges.
    """
    ctnr_graph: nx.Graph = pk.get_container_call_digraph().to_undirected()
    service_graph: nx.Graph = pk.get_service_call_digraph().to_undirected()
    node_ctnr_graph: nx.Graph = nx.Graph()  # Here, a node means a host running containers.
    if (nodes_ctnrs := pk.get_nodes_to_containers()):
        for node, ctnrs in nodes_ctnrs.items():
            # TODO: 'nsenter' container should be removed from original dataset.
            for ctnr in [c for c in ctnrs if c != 'nsenter']:
                node_ctnr_graph.add_edge(node, ctnr)

    G: nx.Graph = nx.Graph()
    for u, v in combinations(nodes, 2):
        if u.is_container() and v.is_container():
            if u.comp == v.comp or ctnr_graph.has_edge(u.comp, v.comp):
                continue
        elif u.is_container() and v.is_service():
            u_service: str = pk.get_service_by_container(u.comp)
            if u_service == v.comp or service_graph.has_edge(u_service, v.comp):
                continue
        elif u.is_service() and v.is_container():
            v_service: str = pk.get_service_by_container(v.comp)
            if u.comp == v_service or service_graph.has_edge(u.comp, v_service):
                continue
        elif u.is_service() and v.is_service():
            if u.comp == v.comp or service_graph.has_edge(u.comp, v.comp):
                continue
        elif u.is_node() and v.is_node():
            # each node has no connectivity.
            pass
        elif u.is_node() and v.is_container():
            if node_ctnr_graph.has_edge(u.comp, v.comp):
                continue
        elif u.is_container() and v.is_node():
            if node_ctnr_graph.has_edge(u.comp, v.comp):
                continue
        elif (u.is_node() and v.is_service()):
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
        # TODO: node and middleware metrics
        else:
            raise ValueError(f"'{u}' or '{v}' has unexpected format")
        # use node number because 'pgmpy' package handles only graph nodes consisted with numpy array.
        G.add_edge(u, v)
    return G


def prepare_init_graph(nodes: mn.MetricNodes, pk: PriorKnowledge) -> nx.Graph:
    """Prepare initialized causal graph."""
    init_g = nx.Graph()
    for (u, v) in combinations(nodes, 2):
        init_g.add_edge(u, v)
    RG: nx.Graph = build_subgraph_of_removal_edges(nodes, pk)
    init_g.remove_edges_from(RG.edges())
    return init_g


def fix_edge_direction_based_hieralchy(G: nx.DiGraph, u: mn.MetricNode, v: mn.MetricNode, pk: PriorKnowledge) -> None:
    # Force direction from (container -> service) to (service -> container) in same service
    if u.is_service() and v.is_container():
        # check whether u and v in the same service
        v_service = pk.get_service_by_container(v.comp)
        if u.comp == v_service:
            nx_util.reverse_edge_direction(G, u, v)


def fix_edge_direction_based_network_call(
    G: nx.DiGraph, u: mn.MetricNode, v: mn.MetricNode,
    service_dep_graph: nx.DiGraph,
    container_dep_graph: nx.DiGraph,
    pk: PriorKnowledge,
) -> None:
    # From service to service
    if u.is_service() and v.is_service():
        # If u and v is in the same service, force bi-directed edge.
        if u.comp == v.comp:
            nx_util.set_bidirected_edge(G, u, v)
        if (v.comp not in service_dep_graph[u.comp]) and \
           (u.comp in service_dep_graph[v.comp]):
            nx_util.reverse_edge_direction(G, u, v)

    # From container to container
    if u.is_container() and v.is_container():
        # If u and v is in the same container, force bi-directed edge.
        if u.comp == v.comp:
            nx_util.set_bidirected_edge(G, u, v)
        elif (v.comp not in container_dep_graph[u.comp]) and \
             (u.comp in container_dep_graph[v.comp]):
            nx_util.reverse_edge_direction(G, u, v)

    # From service to container
    if u.is_service() and v.is_container():
        v_service = pk.get_service_by_container(v.comp)
        if (v_service not in service_dep_graph[u.comp]) and \
           (u.comp in service_dep_graph[v_service]):
            nx_util.reverse_edge_direction(G, u, v)

    # From container to service
    if u.is_container() and v.is_service():
        # u_ctnr = u.split('-', maxsplit=1)[1].split('_')[0]
        # v_service = v.split('-', maxsplit=1)[1].split('_')[0]
        u_service = pk.get_service_by_container(u.comp)
        if (v.comp not in service_dep_graph[u_service]) and \
           (u_service in service_dep_graph[v.comp]):
            nx_util.reverse_edge_direction(G, u, v)


def fix_edge_directions_in_causal_graph(
    G: nx.DiGraph,
    pk: PriorKnowledge,
) -> nx.DiGraph:
    """Fix the edge directions in the causal graphs.
    1. Fix directions based on the system hieralchy such as a service and a container
    2. Fix directions based on the network call graph.
    """
    service_dep_graph: nx.DiGraph = pk.get_service_call_digraph().reverse()
    container_dep_graph: nx.DiGraph = pk.get_container_call_digraph().reverse()
    # Traverse the all edges of G via the neighbors
    for u, nbrsdict in G.adjacency():
        nbrs = list(nbrsdict.keys())  # to avoid 'RuntimeError: dictionary changed size during iteration'
        for v in nbrs:
            # u -> v
            fix_edge_direction_based_hieralchy(G, u, v, pk)
            fix_edge_direction_based_network_call(G, u, v, service_dep_graph, container_dep_graph, pk)
    return G


def build_causal_graph_with_pcalg(
    dm: np.ndarray,
    nodes: mn.MetricNodes,
    init_g: nx.Graph,
    pk: PriorKnowledge,
    pc_citest_alpha: float,
    pc_variant: str = '',
    pc_citest: str = 'fisher-z',
) -> nx.DiGraph:
    """
    Build causal graph with PC algorithm.
    """
    init_g = nx.relabel_nodes(init_g, mapping=nodes.node_to_num)
    cm = np.corrcoef(dm.T)
    ci_test = ci_test_fisher_z if pc_citest == 'fisher-z' else pc_citest
    (G, sep_set) = pcalg.estimate_skeleton(
        indep_test_func=ci_test,
        data_matrix=dm,
        alpha=pc_citest_alpha,
        corr_matrix=cm,
        init_graph=init_g,
        method=pc_variant,
    )
    DG: nx.DiGraph = pcalg.estimate_cpdag(skel_graph=G, sep_set=sep_set)
    DG = nx.relabel_nodes(DG, mapping=nodes.num_to_node)
    return fix_edge_directions_in_causal_graph(DG, pk)


def build_causal_graphs_with_pgmpy(
    df: pd.DataFrame,
    pk: PriorKnowledge,
    pc_citest_alpha: float,
    pc_variant: str = 'orig',
    pc_citest: str = 'fisher-z',
) -> nx.DiGraph:
    c = estimators.PC(data=df)
    ci_test = fisher_z if pc_citest == 'fisher-z' else pc_citest
    G: nx.DiGraph = c.estimate(
        variant=pc_variant,
        ci_test=ci_test,
        significance_level=pc_citest_alpha,
        return_type='pdag',
    )
    return fix_edge_directions_in_causal_graph(G, pk)


def find_connected_subgraphs(G: nx.DiGraph, root_labels: tuple[str, ...]) -> tuple[list[nx.DiGraph], list[nx.DiGraph]]:
    """ Find subgraphs connected components.
    """
    root_contained_subg: list[nx.DiGraph] = []
    root_uncontained_subg: list[nx.DiGraph] = []
    root_nodes = [mn.MetricNode(root) for root in root_labels]
    for c in nx.connected_components(G.to_undirected()):
        subg = G.subgraph(c).copy()
        if any([r in c for r in root_nodes]):
            root_contained_subg.append(subg)
        else:
            root_uncontained_subg.append(subg)
    return root_contained_subg, root_uncontained_subg


def remove_nodes_subgraph_uncontained_root(G: nx.DiGraph, root_labels: tuple[str, ...]) -> nx.DiGraph:
    """Find graphs containing root metric node.
    """
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


def run(
    dataset: pd.DataFrame,
    pk: PriorKnowledge,
    **kwargs,
) -> tuple[nx.DiGraph, tuple[list[nx.DiGraph], list[nx.DiGraph]], dict[str, Any]]:
    dataset = filter_by_target_metrics(dataset, pk)
    if not any(label in dataset.columns for label in pk.get_root_metrics()):
        raise ValueError(f"dataset has no root metric node: {pk.get_root_metrics()}")

    building_graph_start: float = time.time()

    nodes: mn.MetricNodes = mn.MetricNodes.from_dataframe(dataset)
    init_g: nx.Graph = prepare_init_graph(nodes, pk)

    if (pc_library := kwargs['pc_library']) == 'pcalg':
        G = build_causal_graph_with_pcalg(
            dataset.to_numpy(), nodes, init_g, pk,
            pc_variant=kwargs['pc_variant'],
            pc_citest=kwargs['pc_citest'],
            pc_citest_alpha=kwargs['pc_citest_alpha'],
        )
    elif pc_library == 'pgmpy':
        G = build_causal_graphs_with_pgmpy(
            dataset, pk,
            pc_variant=kwargs['pc_variant'],
            pc_citest=kwargs['pc_citest'],
            pc_citest_alpha=kwargs['pc_citest_alpha'],
        )
    else:
        raise ValueError(f"pc_library should be pcalg or pgmpy ({pc_library})")

    root_contained_graphs, root_uncontained_graphs = find_connected_subgraphs(G, pk.get_root_metrics())

    building_graph_elapsed: float = time.time() - building_graph_start

    G = remove_nodes_subgraph_uncontained_root(G, pk.get_root_metrics())  # for stats
    stats = {
        'init_graph_nodes_num': init_g.number_of_nodes(),
        'init_graph_edges_num': init_g.number_of_edges(),
        'causal_graph_nodes_num': G.number_of_nodes(),
        'causal_graph_edges_num': G.number_of_edges(),
        'causal_graph_density': nx.density(G),
        'causal_graph_flow_hierarchy': nx.flow_hierarchy(G),
        'building_graph_elapsed_sec': building_graph_elapsed,
    }
    return G, (root_contained_graphs, root_uncontained_graphs), stats
