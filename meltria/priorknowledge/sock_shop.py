from typing import Final

import networkx as nx

TARGET_APP_NAME: Final[str] = "sock-shop"

ROOT_SERVICE: Final[str] = "front-end"
ROOT_CONTAINER: Final[str] = "front-end"
ROOT_METRIC_LABELS: Final[tuple[str, str, str]] = (
    "s-front-end_latency",
    "s-front-end_throughput",
    "s-front-end_errors",
)

ROOT_METRIC_TYPES_AS_RED: Final[dict[str, str]] = {
    "latency": "s-front-end_latency",
    "throughput": "s-front-end_throughput",
    "errors": "s-front-end_errors",
}

SERVICE_CALL_DIGRAPH: Final[nx.DiGraph] = nx.DiGraph(
    [
        ("front-end", "orders"),
        ("front-end", "catalogue"),
        ("front-end", "user"),
        ("front-end", "carts"),
        ("orders", "shipping"),
        ("orders", "payment"),
        ("orders", "user"),
        ("orders", "carts"),
    ]
)

CONTAINER_CALL_DIGRAPH: Final[nx.DiGraph] = nx.DiGraph(
    [
        ("front-end", "orders"),
        ("front-end", "carts"),
        ("front-end", "user"),
        ("front-end", "catalogue"),
        ("front-end", "session-db"),
        ("orders", "shipping"),
        ("orders", "payment"),
        ("orders", "user"),
        ("orders", "carts"),
        ("orders", "orders-db"),
        ("catalogue", "catalogue-db"),
        ("user", "user-db"),
        ("carts", "carts-db"),
        ("shipping", "rabbitmq"),
        ("rabbitmq", "queue-master"),
    ]
)

CONTAINER_CALL_GRAPH: Final[dict[str, list[str]]] = {
    "front-end": ["orders", "carts", "user", "catalogue"],
    "catalogue": ["front-end", "catalogue-db"],
    "catalogue-db": ["catalogue"],
    "orders": ["front-end", "orders-db", "carts", "user", "payment", "shipping"],
    "orders-db": ["orders"],
    "user": ["front-end", "user-db", "orders"],
    "user-db": ["user"],
    "payment": ["orders"],
    "shipping": ["orders", "rabbitmq"],
    "queue-master": ["rabbitmq"],
    "rabbitmq": ["shipping", "queue-master"],
    "carts": ["front-end", "carts-db", "orders"],
    "carts-db": ["carts"],
    "session-db": ["front-end"],
}

# Use list of tuple because of supporting multiple routes
SERVICE_TO_SERVICES: Final[dict[str, list[str]]] = {
    "orders": ["front-end"],
    "carts": ["orders", "front-end"],
    "user": ["orders", "front-end"],
    "catalogue": ["front-end"],
    "payment": ["orders"],
    "shipping": ["orders"],
    "front-end": [],
}

# TODO: wrong call graph?
SERVICE_TO_SERVICE_ROUTES: Final[dict[str, list[tuple[str, ...]]]] = {
    "orders": [("front-end",)],
    "carts": [("orders", "front-end"), ("front-end",)],
    "user": [("orders", "front-end"), ("front-end",)],
    "catalogue": [("front-end",)],
    "payment": [("orders",)],
    "shipping": [("orders",)],
    "front-end": [()],
}

SERVICE_CONTAINERS: Final[dict[str, list[str]]] = {
    "carts": ["carts", "carts-db"],
    "payment": ["payment"],
    "shipping": ["shipping"],
    "front-end": ["front-end"],
    "user": ["user", "user-db"],
    "catalogue": ["catalogue", "catalogue-db"],
    "orders": ["orders", "orders-db"],
}

CONTAINER_TO_SERVICE: Final[dict[str, str]] = {c: s for s, ctnrs in SERVICE_CONTAINERS.items() for c in ctnrs}

CONTAINER_TO_RUNTIME: dict[str, tuple[str, str]] = {
    "carts": ("web", "jvm"),
    "carts-db": ("db", "mongodb"),
    "shipping": ("web", "jvm"),
    "payment": ("web", "jvm"),
    "front-end": ("web", "nodejs"),
    "user": ("web", "jvm"),
    "user-db": ("db", "mongodb"),
    "orders": ("web", "jvm"),
    "orders-db": ("db", "mongodb"),
    "catalogue": ("web", "go"),
    "catalogue-db": ("db", "mongodb"),
    "queue-master": ("web", "jvm"),
    "session-db": ("db", "mysql"),
    "rabbitmq": ("mq", "rabbitmq"),
}

SKIP_CONTAINERS: Final[list[str]] = ["queue-master", "rabbitmq", "session-db"]
SKIP_SERVICES: Final[list[str]] = [""]

DIAGNOSER_TARGET_DATA: Final[dict[str, list[str]]] = {
    "containers": [],  # all
    "services": [],  # all
    "nodes": [],  # all
    # "node_cpu_seconds_total",
    # "node_disk_io_now",
    # "node_filesystem_avail_bytes",
    # "node_memory_MemAvailable_bytes",
    # "node_network_receive_bytes_total",
    # "node_network_transmit_bytes_total",
    "middlewares": [],  # all
}
