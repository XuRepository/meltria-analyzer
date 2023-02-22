from typing import Final

METRIC_TYPE_SERVICES: Final[str] = "services"
METRIC_TYPE_CONTAINERS: Final[str] = "containers"
METRIC_TYPE_NODES: Final[str] = "nodes"
METRIC_TYPE_MIDDLEWARES: Final[str] = "middlewares"

METRIC_TYPE_MAP: Final[list[tuple[str, str]]] = [
    ("c-", METRIC_TYPE_CONTAINERS),
    ("s-", METRIC_TYPE_SERVICES),
    ("m-", METRIC_TYPE_MIDDLEWARES),
    ("n-", METRIC_TYPE_NODES),
]

METRIC_PREFIX_TO_TYPE: Final[dict[str, str]] = dict([(v, k) for k, v in METRIC_TYPE_MAP])

ALL_METRIC_TYPES: Final[dict[str, bool]] = {
    METRIC_TYPE_SERVICES: True,
    METRIC_TYPE_NODES: True,
    METRIC_TYPE_CONTAINERS: True,
    METRIC_TYPE_MIDDLEWARES: True,
}
