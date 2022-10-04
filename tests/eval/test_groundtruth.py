import networkx as nx
import pytest

from diagnoser import metric_node as mn
from eval import groundtruth
from meltria.loader import METRIC_TYPE_CONTAINERS, METRIC_TYPE_MIDDLEWARES, METRIC_TYPE_NODES, METRIC_TYPE_SERVICES
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
ss_pk_with_middlewares = new_knowledge(
    "sock-shop",
    {
        METRIC_TYPE_CONTAINERS: True,
        METRIC_TYPE_SERVICES: True,
        METRIC_TYPE_MIDDLEWARES: True,
        METRIC_TYPE_NODES: False,
    },
    {"node-containers": {}},
)
ts_pk_without_middlewares = new_knowledge(
    "train-ticket",
    {
        METRIC_TYPE_CONTAINERS: True,
        METRIC_TYPE_SERVICES: True,
        METRIC_TYPE_MIDDLEWARES: False,
        METRIC_TYPE_NODES: False,
    },
    {"node-containers": {}},
)
ts_pk_with_middlewares = new_knowledge(
    "train-ticket",
    {
        METRIC_TYPE_CONTAINERS: True,
        METRIC_TYPE_SERVICES: True,
        METRIC_TYPE_MIDDLEWARES: True,
        METRIC_TYPE_NODES: False,
    },
    {"node-containers": {}},
)


def test_select_ground_truth_metrics_in_routes() -> None:
    metrics = [
        "c-ts-station-service_cpu_system_seconds_total",
        "c-ts-station-service_sockets",
        "s-ts-station_request_duration_seconds",
        "s-ts-order_request_duration_seconds",
        "s-ts-ui-dashboard_requests_count",
        "s-ts-ui-dashboard_request_errors_count",
        "s-ts-ui-dashboard_request_duration_seconds",
    ]
    routes = groundtruth.select_ground_truth_metrics_in_routes(
        ts_pk_without_middlewares, metrics, "pod-cpu-hog", "ts-station-service"
    )
    expected: list[list[str]] = [
        [
            "c-ts-station-service_cpu_system_seconds_total",
            "c-ts-station-service_sockets",
            "s-ts-station_request_duration_seconds",
            "s-ts-ui-dashboard_requests_count",
            "s-ts-ui-dashboard_request_errors_count",
            "s-ts-ui-dashboard_request_duration_seconds",
        ],
        [
            "c-ts-station-service_cpu_system_seconds_total",
            "c-ts-station-service_sockets",
            "s-ts-station_request_duration_seconds",
            "s-ts-order_request_duration_seconds",
            "s-ts-ui-dashboard_requests_count",
            "s-ts-ui-dashboard_request_errors_count",
            "s-ts-ui-dashboard_request_duration_seconds",
        ],
    ]
    assert [r[0] for r in routes] == expected


def test_check_tsdr_ground_truth_by_route() -> None:
    metrics = [
        "c-user-db_cpu_usage_seconds_total",
        "c-user-db_cpu_user_seconds_total",
        "c-user-db_network_receive_bytes",
        "s-user_latency",
        "c-orders_cpu_usage_seconds_total",
        "s-orders_latency",
        "s-front-end_latency",
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
        ss_pk_without_middlewares, metrics, "pod-cpu-hog", "user-db"
    )
    assert ok is True
    expected = [
        "c-user-db_cpu_usage_seconds_total",
        "c-user-db_cpu_user_seconds_total",
        "s-user_latency",
        "s-orders_latency",
        "s-front-end_latency",
    ]
    assert found_metrics == expected

    # without orders
    metrics = [
        "c-user-db_cpu_usage_seconds_total",
        "c-user-db_cpu_user_seconds_total",
        "c-user-db_network_receive_bytes",
        "s-user_latency",
        "s-front-end_latency",
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
        ss_pk_without_middlewares, metrics, "pod-cpu-hog", "user-db"
    )
    assert ok is True
    expected = [
        "c-user-db_cpu_usage_seconds_total",
        "c-user-db_cpu_user_seconds_total",
        "s-user_latency",
        "s-front-end_latency",
    ]
    assert found_metrics == expected

    # not match
    metrics = [
        "c-user-db_cpu_usage_seconds_total",
        "c-user-db_cpu_user_seconds_total",
        "c-user-db_network_receive_bytes",
        "s-user_latency",
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
        ss_pk_without_middlewares, metrics, "pod-cpu-hog", "user-db"
    )
    assert ok is False
    expected = [
        "c-user-db_cpu_usage_seconds_total",
        "c-user-db_cpu_user_seconds_total",
        "s-user_latency",
    ]
    assert found_metrics == expected

    # only front-end
    metrics = [
        "c-front-end_cpu_usage_seconds_total",
        "c-front-end_cpu_user_seconds_total",
        "c-user-db_network_receive_bytes",
        "s-front-end_latency",
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
        ss_pk_without_middlewares, metrics, "pod-cpu-hog", "front-end"
    )
    assert ok is True
    expected = [
        "c-front-end_cpu_usage_seconds_total",
        "c-front-end_cpu_user_seconds_total",
        "s-front-end_latency",
    ]
    assert found_metrics == expected

    # with middlewares
    metrics = [
        "c-user-db_cpu_usage_seconds_total",
        "c-user-db_network_receive_bytes",
        "m-user-db_mongodb_sys_cpu_user_ms",
        "m-user-db_mongodb_sys_cpu_idle_ms",
        "s-user_latency",
        "c-orders_cpu_usage_seconds_total",
        "s-orders_latency",
        "s-front-end_latency",
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
        ss_pk_with_middlewares, metrics, "pod-cpu-hog", "user-db"
    )
    assert ok is True
    expected = [
        "c-user-db_cpu_usage_seconds_total",
        "m-user-db_mongodb_sys_cpu_user_ms",
        "m-user-db_mongodb_sys_cpu_idle_ms",
        "s-user_latency",
        "s-orders_latency",
        "s-front-end_latency",
    ]
    assert found_metrics == expected

    # no input of middleware metrics with middlewares
    metrics = [
        "c-user-db_cpu_usage_seconds_total",
        "c-user-db_network_receive_bytes",
        "s-user_latency",
        "c-orders_cpu_usage_seconds_total",
        "s-orders_latency",
        "s-front-end_latency",
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
        ss_pk_with_middlewares, metrics, "pod-cpu-hog", "user-db"
    )
    assert ok is False
    expected = [
        "c-user-db_cpu_usage_seconds_total",
        "s-user_latency",
        "s-orders_latency",
        "s-front-end_latency",
    ]
    assert found_metrics == expected


def test_check_tsdr_ground_truth_by_route_train_ticket() -> None:
    metrics = [
        "c-ts-food-mongo_cpu_usage_seconds_total",
        "c-ts-food-mongo_cpu_user_seconds_total",
        "c-ts-food_mongo_network_receive_bytes",
        "s-ts-food_request_duration_seconds",
        "c-ts-preserve_cpu_usage_seconds_total",
        "s-ts-preserve_request_duration_seconds",
        "s-ts-ui-dashboard_request_duration_seconds",
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
        ts_pk_without_middlewares, metrics, "pod-cpu-hog", "ts-food-mongo"
    )
    assert ok is True
    expected = [
        "c-ts-food-mongo_cpu_usage_seconds_total",
        "c-ts-food-mongo_cpu_user_seconds_total",
        "s-ts-food_request_duration_seconds",
        "s-ts-preserve_request_duration_seconds",
        "s-ts-ui-dashboard_request_duration_seconds",
    ]
    assert found_metrics == expected


@pytest.mark.parametrize(
    "desc,chaos_type,chaos_comp,input,expected",
    [
        (
            "correct01",
            "pod-cpu-hog",
            "user-db",
            [
                # u (cause) -> v
                ("s-user_latency", "s-front-end_latency"),
                ("c-user-db_cpu_usage_seconds_total", "s-user_latency"),
                ("s-orders_latency", "s-front-end_latency"),
                ("c-orders-db_cpu_usage_seconds_total", "s-orders_latency"),
                ("s-user_latency", "s-orders_latency"),
            ],
            [
                [
                    "s-front-end_latency",
                    "s-orders_latency",
                    "s-user_latency",
                    "c-user-db_cpu_usage_seconds_total",
                ],
                [
                    "s-front-end_latency",
                    "s-user_latency",
                    "c-user-db_cpu_usage_seconds_total",
                ],
            ],
        ),
        (
            "correct02: cause container metric -> the other service metric",
            "pod-cpu-hog",
            "user-db",
            [
                # u (cause) -> v
                ("c-user-db_cpu_usage_seconds_total", "s-front-end_latency"),
            ],
            [
                ["s-front-end_latency", "c-user-db_cpu_usage_seconds_total"],
            ],
        ),
        (
            "correct03: cause container metric -> container metric",
            "pod-cpu-hog",
            "user-db",
            [
                # u (cause) -> v
                ("c-user-db_cpu_usage_seconds_total", "s-front-end_latency"),
                (
                    "c-user-db_cpu_usage_seconds_total",
                    "c-user-db_cpu_user_seconds_total",
                ),
                ("s-orders_latency", "s-front-end_latency"),
                ("c-user-db_cpu_user_seconds_total", "s-orders_latency"),
            ],
            [
                [
                    "s-front-end_latency",
                    "s-orders_latency",
                    "c-user-db_cpu_user_seconds_total",
                    "c-user-db_cpu_usage_seconds_total",
                ],
                [
                    "s-front-end_latency",
                    "s-orders_latency",
                    "c-user-db_cpu_user_seconds_total",
                ],
                ["s-front-end_latency", "c-user-db_cpu_usage_seconds_total"],
            ],
        ),
        (
            "correct04: bi-directed graph",
            "pod-cpu-hog",
            "carts",
            [
                ["c-carts_cpu_user_seconds_total", "s-front-end_latency"],
                ["s-carts_latency", "s-front-end_latency"],
                ["c-carts_threads", "s-carts_latency"],
                ["c-carts_threads", "c-carts_cpu_user_seconds_total"],
                ["c-carts_cpu_user_seconds_total", "c-carts_threads"],
            ],
            [
                ["s-front-end_latency", "c-carts_cpu_user_seconds_total"],
                [
                    "s-front-end_latency",
                    "c-carts_cpu_user_seconds_total",
                    "c-carts_threads",
                ],
                ["s-front-end_latency", "s-carts_latency", "c-carts_threads"],
                [
                    "s-front-end_latency",
                    "s-carts_latency",
                    "c-carts_threads",
                    "c-carts_cpu_user_seconds_total",
                ],
            ],
        ),
    ],
    ids=["correct01", "correct02", "correct03", "correct04"],
)
def test_check_causal_graph(desc, chaos_type, chaos_comp, input, expected) -> None:
    edges = [(mn.MetricNode(u), mn.MetricNode(v)) for (u, v) in input]
    expected_edges = [[mn.MetricNode(n) for n in path] for path in expected]
    G = nx.DiGraph(edges)
    ok, routes = groundtruth.check_causal_graph(ss_pk_without_middlewares, G, chaos_type, chaos_comp)
    assert ok
    assert sorted([r.nodes for r in routes]) == sorted(expected_edges)
