from itertools import combinations

import networkx as nx
import pandas as pd

import diagnoser.metric_node as mn
from diagnoser import nx_util
from meltria.priorknowledge.priorknowledge import PriorKnowledge


def get_forbits(dataset: pd.DataFrame, pk: PriorKnowledge) -> list[tuple[str, str]]:
    nodes: mn.MetricNodes = mn.MetricNodes.from_dataframe(dataset)
    init_g: nx.Graph = prepare_init_graph(
        nodes, pk, enable_prior_knowledge=True,
    )
    init_dg = fix_edge_directions_in_causal_graph(init_g, pk)
    forbits = []
    for node1, node2 in combinations(init_dg.nodes, 2):
        if not init_dg.has_edge(node1, node2):
            forbits.append((node1.label, node2.label))
        if not init_dg.has_edge(node2, node1):
            forbits.append((node2.label, node1.label))
    return forbits


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
