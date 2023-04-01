from itertools import permutations

import networkx as nx


def estimate_skeleton_with_indep_test(indep_test_func, data_matrix, alpha, **kwargs) -> tuple[nx.Graph, list[set]]:
    """Estimate a skeleton graph from the statistis information.

    Args:
        indep_test_func: the function name for a conditional
            independency test.
        data_matrix: data (as a numpy array).
        alpha: the significance level.
        kwargs:
            'init_graph': initial structure of skeleton graph
                (as a networkx.Graph). If not specified,
                a complete graph is used.
            'fixed_edges': Undirected edges marked here are not changed
                (as a networkx.Graph). If not specified,
                an empty graph is used.
            other parameters may be passed depending on the
                indep_test_func()s.
    Returns:
        g: a skeleton graph (as a networkx.Graph).

    [Colombo2014] Diego Colombo and Marloes H Maathuis. Order-independent
    constraint-based causal structure learning. In The Journal of Machine
    Learning Research, Vol. 15, pp. 3741-3782, 2014.
    """

    node_ids = range(data_matrix.shape[1])
    if "init_graph" in kwargs:
        g = kwargs["init_graph"]
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError("init_graph not matching data_matrix shape")
    else:
        g = nx.complete_graph(node_ids)

    fixed_edges = set()
    if "fixed_edges" in kwargs:
        _fixed_edges = kwargs["fixed_edges"]
        if not isinstance(_fixed_edges, nx.Graph):
            raise ValueError
        if not _fixed_edges.number_of_nodes() == len(node_ids):
            raise ValueError("fixed_edges not matching data_matrix shape")
        for (i, j) in _fixed_edges.edges:
            fixed_edges.add((i, j))
            fixed_edges.add((j, i))

    for (i, j) in permutations(node_ids, 2):
        if (i, j) in fixed_edges:
            continue

        p_val = indep_test_func(data_matrix, i, j, set(), **kwargs)
        if (not kwargs["disable_ci_edge_cut"]) and p_val > alpha:
            if g.has_edge(i, j):
                g.remove_edge(i, j)
            break
        if p_val == 0.0:
            p_val = 1e-10
        # g[i][j]["weight"] = 1 / p_val

    return g
