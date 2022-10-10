from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cache

import networkx as nx

from meltria.priorknowledge import sock_shop, train_ticket


class PriorKnowledge(ABC):
    def __init__(self, mappings: dict[str, dict[str, list[str]]], target_metric_types: dict[str, bool]) -> None:
        self.mappings = mappings  # mappings includes node to/from container mapping .
        self.target_metric_types = target_metric_types

    @abstractmethod
    def get_root_service(self) -> str:
        pass

    @abstractmethod
    def get_root_container(self) -> str:
        pass

    @abstractmethod
    def get_containers(self, skip: bool = False) -> list[str]:
        pass

    @abstractmethod
    def get_root_metrics(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def get_service_call_digraph(self) -> nx.DiGraph:
        pass

    @abstractmethod
    def get_container_call_digraph(self) -> nx.DiGraph:
        pass

    @abstractmethod
    def get_container_call_graph(self, ctnr: str) -> list[str]:
        pass

    def get_container_neighbors_in_service(self, ctnr: str) -> list[str]:
        neighbors: list[str] = []
        service: str | None = self.get_service_by_container(ctnr)
        for neighbor in self.get_container_call_graph(ctnr):
            if service == self.get_service_by_container(neighbor):
                neighbors.append(neighbor)
        return neighbors

    @abstractmethod
    def get_service_routes(self, service: str) -> list[tuple[str, ...]]:
        pass

    @abstractmethod
    def get_containers_of_service(self) -> dict[str, list[str]]:
        pass

    @abstractmethod
    def get_service_containers(self, service: str) -> list[str]:
        pass

    @abstractmethod
    def get_service_by_container(self, ctnr: str) -> str:
        pass

    @abstractmethod
    def get_role_and_runtime_by_container(self, ctnr: str) -> tuple[str, str]:
        pass

    @abstractmethod
    def get_skip_containers(self) -> list[str]:
        pass

    @abstractmethod
    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        pass

    def get_nodes_to_containers(self) -> dict[str, list[str]]:
        """Example
        "nodes-containers": {
            "gke-train-ticket-01-default-pool-1db6151d-0fxo": [
                "ts-assurance-mongo",
                "ts-verification-code-service",
                "ts-consign-price-mongo",
                "ts-train-mongo",
                "ts-order-service",
                "ts-station-mongo",
                "ts-food-map-mongo",
                "ts-notification-service"
            ],
        }
        """
        return self.mappings.get("nodes-containers")

    def get_nodes(self) -> list[str]:
        return list(self.get_nodes_to_containers().keys())

    def get_nodes_to_containers_graph(self) -> nx.Graph:
        G: nx.Graph = nx.Graph()  # Here, a node means a host running containers.
        if nodes_ctnrs := self.get_nodes_to_containers():
            for node, ctnrs in nodes_ctnrs.items():
                # 'nsenter' container should be removed from original dataset.
                for ctnr in [c for c in ctnrs if c != "nsenter"]:
                    G.add_edge(node, ctnr)
        return G

    def group_metrics_by_service(self, metrics: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(lambda: list())
        for metric in metrics:
            # TODO: resolve duplicated code of MetricNode class.
            comp, base_name = metric.split("-", maxsplit=1)[1].split("_", maxsplit=1)
            if metric.startswith("c-"):
                service = self.get_service_by_container(comp)
            elif metric.startswith("s-"):
                service = comp
            elif metric.startswith("m-"):
                service = self.get_service_by_container(comp)
            else:
                raise ValueError(f"{metric} is invalid")
            groups[service].append(metric)
        return groups

    def is_target_metric_type(self, metric_type: str) -> bool:
        assert metric_type in self.target_metric_types, f"{metric_type} is not defined in target_metric_types"
        return self.target_metric_types[metric_type]

    @staticmethod
    @cache
    def _generate_service_to_service_routes(
        service_call_g: nx.DiGraph,
        root_service: str,
    ) -> dict[str, list[tuple[str, ...]]]:
        """Generate adjacency list of service to service routes."""
        stos_routes: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        nodes = [n for n in service_call_g.nodes if n not in root_service]
        paths = nx.all_simple_paths(service_call_g, source=root_service, target=nodes)
        for path in paths:
            path.reverse()
            source_service = path[0]
            stos_routes[source_service].append(tuple(path[1:]))
        stos_routes[root_service] = [tuple([root_service])]
        return stos_routes


class SockShopKnowledge(PriorKnowledge):
    def __init__(self, mappings: dict[str, dict[str, list[str]]], target_metric_types: dict[str, bool]) -> None:
        super().__init__(mappings, target_metric_types)

    def get_root_service(self) -> str:
        return sock_shop.ROOT_SERVICE

    def get_root_container(self) -> str:
        return sock_shop.ROOT_CONTAINER

    def get_containers(self, skip: bool = False) -> list[str]:
        ctnrs = list(sock_shop.CONTAINER_CALL_GRAPH.keys())
        if skip:
            return [ctnr for ctnr in ctnrs if ctnr not in self.get_skip_containers()]
        return ctnrs

    def get_root_metrics(self) -> tuple[str, ...]:
        return sock_shop.ROOT_METRIC_LABELS

    def get_service_call_digraph(self) -> nx.DiGraph:
        return sock_shop.SERVICE_CALL_DIGRAPH

    def get_container_call_digraph(self) -> nx.DiGraph:
        return sock_shop.CONTAINER_CALL_DIGRAPH

    def get_container_call_graph(self, ctnr: str) -> list[str]:
        assert ctnr in sock_shop.CONTAINER_CALL_GRAPH, f"{ctnr} is not defined in container_call_graph"
        return sock_shop.CONTAINER_CALL_GRAPH[ctnr]

    def get_service_routes(self, service: str) -> list[tuple[str, ...]]:
        routes = self._generate_service_to_service_routes(sock_shop.SERVICE_CALL_DIGRAPH, sock_shop.ROOT_SERVICE)
        return routes[service]

    def get_containers_of_service(self) -> dict[str, list[str]]:
        return sock_shop.SERVICE_CONTAINERS

    def get_service_containers(self, service: str) -> list[str]:
        assert service in sock_shop.SERVICE_CONTAINERS, f"{service} is not defined in service_containers"
        return sock_shop.SERVICE_CONTAINERS[service]

    def get_service_by_container(self, ctnr: str) -> str:
        assert ctnr in sock_shop.CONTAINER_TO_SERVICE, f"{ctnr} is not defined in container_service"
        return sock_shop.CONTAINER_TO_SERVICE[ctnr]

    def get_role_and_runtime_by_container(self, ctnr: str) -> tuple[str, str]:
        assert ctnr in sock_shop.CONTAINER_TO_RUNTIME, f"{ctnr} is not defined in container_role_runtime"
        return sock_shop.CONTAINER_TO_RUNTIME[ctnr]

    def get_skip_containers(self) -> list[str]:
        return sock_shop.SKIP_CONTAINERS

    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        return sock_shop.DIAGNOSER_TARGET_DATA


class TrainTicketKnowledge(PriorKnowledge):
    def __init__(self, mappings: dict[str, dict[str, list[str]]], target_metric_types: dict[str, bool]) -> None:
        super().__init__(mappings, target_metric_types)

    def get_root_service(self) -> str:
        return train_ticket.ROOT_SERVICE

    def get_root_container(self) -> str:
        return train_ticket.ROOT_CONTAINER

    def get_containers(self, skip: bool = False) -> list[str]:
        ctnrs = list(train_ticket.CONTAINER_CALL_GRAPH.keys())
        if skip:
            return [ctnr for ctnr in ctnrs if ctnr not in self.get_skip_containers()]
        return ctnrs

    def get_root_metrics(self) -> tuple[str, ...]:
        return train_ticket.ROOT_METRIC_LABELS

    def get_service_call_digraph(self) -> nx.DiGraph:
        return train_ticket.SERVICE_CALL_DIGRAPH

    def get_container_call_digraph(self) -> nx.DiGraph:
        return train_ticket.CONTAINER_CALL_DIGRAPH

    def get_container_call_graph(self, ctnr: str) -> list[str]:
        assert ctnr in train_ticket.CONTAINER_CALL_GRAPH, f"{ctnr} is not defined in container_call_graph"
        return train_ticket.CONTAINER_CALL_GRAPH[ctnr]

    def get_service_routes(self, service: str) -> list[tuple[str, ...]]:
        routes = self._generate_service_to_service_routes(train_ticket.SERVICE_CALL_DIGRAPH, train_ticket.ROOT_SERVICE)
        assert service in routes, f"{service} is not defined in service_call_graph"
        return routes[service]

    def get_containers_of_service(self) -> dict[str, list[str]]:
        return train_ticket.SERVICE_CONTAINERS

    def get_service_containers(self, service: str) -> list[str]:
        assert service in train_ticket.SERVICE_CONTAINERS, f"{service} is not defined in service_containers"
        return train_ticket.SERVICE_CONTAINERS[service]

    def get_service_by_container(self, ctnr: str) -> str:
        assert ctnr in train_ticket.CONTAINER_TO_SERVICE, f"{ctnr} is not defined in container_service"
        return train_ticket.CONTAINER_TO_SERVICE[ctnr]

    def get_skip_containers(self) -> list[str]:
        return train_ticket.SKIP_CONTAINERS

    def get_role_and_runtime_by_container(self, ctnr: str) -> tuple[str, str]:
        ctnr_runtime = train_ticket.generate_container_runtime()
        assert ctnr in ctnr_runtime, f"{ctnr} is not defined in container_runtime"
        return ctnr_runtime[ctnr]

    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        return train_ticket.DIAGNOSER_TARGET_DATA


def new_knowledge(
    target_app: str, target_metric_types: dict[str, bool], mappings: dict[str, dict[str, list[str]]]
) -> PriorKnowledge:
    """Create new knowledge object for the given target app."""
    match target_app:
        case sock_shop.TARGET_APP_NAME:
            return SockShopKnowledge(mappings, target_metric_types)
        case train_ticket.TARGET_APP_NAME:
            return TrainTicketKnowledge(mappings, target_metric_types)
        case _:
            raise ValueError(f"{target_app} is invalid")
