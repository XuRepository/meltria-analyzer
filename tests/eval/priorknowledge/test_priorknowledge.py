from meltria.metric_types import (
    METRIC_TYPE_CONTAINERS,
    METRIC_TYPE_MIDDLEWARES,
    METRIC_TYPE_NODES,
    METRIC_TYPE_SERVICES,
)
from meltria.priorknowledge.priorknowledge import new_knowledge

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


def test_generate_service_to_service_routes() -> None:
    res = ts_pk_without_middlewares.get_service_routes("ts-route-plan")
    assert res == [("ts-travel-plan", "ts-ui-dashboard"), ("ts-ui-dashboard",)]
    res = ts_pk_without_middlewares.get_service_routes("ts-avatar")
    assert res == [("ts-ui-dashboard",)]
