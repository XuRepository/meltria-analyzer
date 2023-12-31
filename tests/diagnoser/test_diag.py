import networkx as nx
import pytest

import diagnoser.metric_node as mn
from diagnoser import call_graph, diag
from meltria.metric_types import (METRIC_TYPE_CONTAINERS,
                                  METRIC_TYPE_MIDDLEWARES, METRIC_TYPE_NODES,
                                  METRIC_TYPE_SERVICES)
from meltria.priorknowledge.priorknowledge import new_knowledge

ss_pk_without_middlewares = new_knowledge(
    "sock-shop",
    {
        METRIC_TYPE_CONTAINERS: True,
        METRIC_TYPE_SERVICES: True,
        METRIC_TYPE_MIDDLEWARES: False,
        METRIC_TYPE_NODES: False,
    },
    {"node-containers": {}},
)


def test_build_subgraph_of_removal_edges():
    metrics = [
        "s-front-end_latency",
        "s-orders_latency",
        "c-orders_sockets",
        "c-orders-db_cpu_usage_seconds_total",
        "s-user_latency",
        "c-user_sockets",
        "c-user_cpu_usage_seconds_total",
        "c-user-db_cpu_usage_seconds_total",
        "n-gke-test-default-pool-66a015a8-9pw7_cpu_seconds_total",
        "n-gke-test-default-pool-1dda290g-n10b_cpu_seconds_total",
    ]
    nodes: mn.MetricNodes = mn.MetricNodes({i: mn.MetricNode(v) for i, v in enumerate(metrics)})
    pk = new_knowledge(
        "sock-shop",
        {
            METRIC_TYPE_CONTAINERS: True,
            METRIC_TYPE_SERVICES: True,
            METRIC_TYPE_MIDDLEWARES: False,
            METRIC_TYPE_NODES: True,
        },
        {
            "nodes-containers": {
                "gke-test-default-pool-66a015a8-9pw7": [
                    "user",
                    "front-end",
                    "orders-db",
                ],
                "gke-test-default-pool-1dda290g-n10b": ["user-db", "orders"],
            }
        },
    )
    RG: nx.Graph = call_graph.build_subgraph_of_removal_edges(nodes, pk)
    expected = [
        ("c-orders_sockets", "c-user-db_cpu_usage_seconds_total"),
        ("c-orders_sockets", "n-gke-test-default-pool-66a015a8-9pw7_cpu_seconds_total"),
        ("c-user-db_cpu_usage_seconds_total", "c-orders-db_cpu_usage_seconds_total"),
        ("c-orders-db_cpu_usage_seconds_total", "c-user_sockets"),
        ("c-orders-db_cpu_usage_seconds_total", "c-user_cpu_usage_seconds_total"),
        (
            "c-user-db_cpu_usage_seconds_total",
            "n-gke-test-default-pool-66a015a8-9pw7_cpu_seconds_total",
        ),
        (
            "n-gke-test-default-pool-1dda290g-n10b_cpu_seconds_total",
            "c-orders-db_cpu_usage_seconds_total",
        ),
        (
            "n-gke-test-default-pool-1dda290g-n10b_cpu_seconds_total",
            "c-user_cpu_usage_seconds_total",
        ),
        ("n-gke-test-default-pool-1dda290g-n10b_cpu_seconds_total", "c-user_sockets"),
        (
            "n-gke-test-default-pool-1dda290g-n10b_cpu_seconds_total",
            "n-gke-test-default-pool-66a015a8-9pw7_cpu_seconds_total",
        ),
        (
            "s-front-end_latency",
            "n-gke-test-default-pool-1dda290g-n10b_cpu_seconds_total",
        ),
    ]
    assert sorted([(u.label, v.label) for (u, v) in list(RG.edges)]) == sorted(expected)


@pytest.mark.parametrize(
    "case,input,expected",
    [
        (
            "hieralchy01: reverse single direction edge",
            [
                ("s-user_latency", "s-front-end_latency"),
                ("s-user_latency", "c-user_cpu_usage_seconds_total"),  # wrong
            ],
            [
                ("s-user_latency", "s-front-end_latency", {}),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
            ],
        ),
        (
            "hieralchy02: determine bi-direction edge",
            [
                ("s-user_latency", "s-front-end_latency"),
                ("s-user_latency", "c-user_cpu_usage_seconds_total"),
                ("c-user_cpu_usage_seconds_total", "s-user_latency"),
            ],
            [
                ("s-user_latency", "s-front-end_latency", {}),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
            ],
        ),
        (
            "nwcall01: service to service",
            [
                ("s-user_latency", "s-front-end_latency"),
                ("s-front-end_latency", "s-user_latency"),  # wrong
                ("s-user_latency", "s-orders_throughput"),
            ],
            [
                ("s-user_latency", "s-front-end_latency", {}),
                ("s-user_latency", "s-orders_throughput", {}),
            ],
        ),
        (
            "nwcall02: container to container",
            [
                (
                    "c-user_cpu_usage_seconds_total",
                    "c-user-db_cpu_usage_seconds_total",
                ),  # wrong
                ("c-user_cpu_usage_seconds_total", "s-user_latency"),
            ],
            [
                (
                    "c-user-db_cpu_usage_seconds_total",
                    "c-user_cpu_usage_seconds_total",
                    {},
                ),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
            ],
        ),
        (
            "nwcall03: service to container",
            [
                ("s-user_latency", "c-user_cpu_usage_seconds_total"),  # wrong
            ],
            [
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
            ],
        ),
        (
            "nwcall04: container to service",
            [
                ("c-orders_cpu_usage_seconds_total", "s-user_latency"),  # wrong
            ],
            [
                ("s-user_latency", "c-orders_cpu_usage_seconds_total", {}),
            ],
        ),
        (
            "nwcall05: container to container or service to service in the same container or service",
            [
                ("s-user_throughput", "s-user_latency"),  # wrong
                ("c-user_cpu_usage_seconds_total", "s-user_latency"),
                (
                    "c-user_cpu_usage_seconds_total",
                    "c-user_memory_working_set_bytes",
                ),  # wrong
            ],
            [
                ("s-user_throughput", "s-user_latency", {}),
                ("s-user_latency", "s-user_throughput", {}),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
                (
                    "c-user_cpu_usage_seconds_total",
                    "c-user_memory_working_set_bytes",
                    {},
                ),
                (
                    "c-user_memory_working_set_bytes",
                    "c-user_cpu_usage_seconds_total",
                    {},
                ),
            ],
        ),
        (
            "hybrid01: mixed hieralchy and nwcall",
            [
                ("s-user_latency", "s-front-end_latency"),
                ("s-user_latency", "c-user_cpu_usage_seconds_total"),  # wrong
                (
                    "c-user_cpu_usage_seconds_total",
                    "c-user-db_cpu_usage_seconds_total",
                ),  # wrong
                ("c-user_cpu_usage_seconds_total", "s-user_latency"),
            ],
            [
                ("s-user_latency", "s-front-end_latency", {}),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
                (
                    "c-user-db_cpu_usage_seconds_total",
                    "c-user_cpu_usage_seconds_total",
                    {},
                ),
            ],
        ),
    ],
    ids=[
        "hieralchy01",
        "hieralchy02",
        "nwcall01",
        "nwcall02",
        "nwcall03",
        "nwcall04",
        "nwcall05",
        "hybrid01",
    ],
)
def test_fix_edge_directions_in_causal_graph(case, input, expected):
    G = nx.DiGraph()
    paths = [(mn.MetricNode(u), mn.MetricNode(v)) for u, v in input]
    G.add_edges_from(paths)
    got = call_graph.fix_edge_directions_in_causal_graph(G, ss_pk_without_middlewares)
    assert sorted([(u.label, v.label, {}) for u, v in got.edges]) == sorted(expected)


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            [
                ("s-user_latency", "s-front-end_latency"),
                ("c-user_cpu_usage_seconds_total", "s-user_latency"),
                ("c-orders_sockets", "s-orders_latency"),
                ("c-orders-db_sockets", "c-orders_sockets"),
            ],
            (
                [
                    [
                        ("s-user_latency", "s-front-end_latency", {}),
                        ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
                    ],
                ],
                [
                    [
                        ("c-orders_sockets", "s-orders_latency", {}),
                        ("c-orders-db_sockets", "c-orders_sockets", {}),
                    ],
                ],
            ),
        ),
    ],
)
def test_find_connected_subgraphs(input, expected):
    G = nx.DiGraph()
    G.add_edges_from([(mn.MetricNode(u), mn.MetricNode(v)) for u, v in input])
    root_contained_subgs, root_uncontained_subgs = diag.find_connected_subgraphs(
        G, ss_pk_without_middlewares.get_root_metrics()
    )

    assert sorted([(u.label, v.label, {}) for u, v in root_contained_subgs[0].edges]) == sorted(expected[0][0])
    assert sorted([(u.label, v.label, {}) for u, v in root_uncontained_subgs[0].edges]) == sorted(expected[1][0])
