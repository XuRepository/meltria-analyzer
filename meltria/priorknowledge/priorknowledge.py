from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cache

import networkx as nx

from meltria.priorknowledge import sock_shop, train_ticket


class PriorKnowledge(ABC):
    @abstractmethod
    def get_root_service(self):
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

    @abstractmethod
    def get_service_routes(self, service: str) -> list[tuple[str, ...]]:
        pass

    @abstractmethod
    def get_containers_of_service(self) -> dict[str, list[str]]:
        pass

    @abstractmethod
    def get_service_containers(self, service: str) -> list[str] | None:
        pass

    @abstractmethod
    def get_service_by_container(self, ctnr: str) -> str | None:
        pass

    @abstractmethod
    def get_skip_containers(self) -> list[str]:
        pass

    @abstractmethod
    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        pass

    def group_metrics_by_service(self, metrics: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(lambda: list())
        for metric in metrics:
            # TODO: resolve duplicated code of MetricNode class.
            comp, base_name = metric.split('-', maxsplit=1)[1].split('_', maxsplit=1)
            if metric.startswith('c-'):
                service = self.get_service_by_container(comp)
            elif metric.startswith('s-'):
                service = comp
            elif metric.startswith('m-'):
                service = self.get_service_by_container(comp)
            else:
                raise ValueError(f'{metric} is invalid')
            groups[service].append(metric)
        return groups

    @staticmethod
    @cache
    def _generate_service_to_service_routes(
        service_call_g: nx.DiGraph, root_service: str,
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
    def __init__(self) -> None:
        pass

    def get_root_service(self):
        return sock_shop.ROOT_SERVICE

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

    def get_container_call_graph(self, ctnr: str) -> list[str] | None:
        return sock_shop.CONTAINER_CALL_GRAPH.get(ctnr)

    def get_service_routes(self, service: str) -> list[tuple[str, ...]]:
        routes = self._generate_service_to_service_routes(sock_shop.SERVICE_CALL_DIGRAPH, sock_shop.ROOT_SERVICE)
        return routes[service]

    def get_containers_of_service(self) -> dict[str, list[str]]:
        return sock_shop.SERVICE_CONTAINERS

    def get_service_containers(self, service: str) -> list[str] | None:
        return sock_shop.SERVICE_CONTAINERS.get(service)

    def get_service_by_container(self, ctnr: str) -> str | None:
        return sock_shop.CONTAINER_TO_SERVICE.get(ctnr)

    def get_skip_containers(self) -> list[str]:
        return sock_shop.SKIP_CONTAINERS

    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        return sock_shop.DIAGNOSER_TARGET_DATA


class TrainTicketKnowledge(PriorKnowledge):
    def __init__(self) -> None:
        pass

    def get_root_service(self):
        return train_ticket.ROOT_SERVICE

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
        return train_ticket.CONTAINER_CALL_GRAPH[ctnr]

    def get_service_routes(self, service: str) -> list[tuple[str, ...]]:
        routes = self._generate_service_to_service_routes(train_ticket.SERVICE_CALL_DIGRAPH, train_ticket.ROOT_SERVICE)
        return routes[service]

    def get_containers_of_service(self) -> dict[str, list[str]]:
        return train_ticket.SERVICE_CONTAINERS

    def get_service_containers(self, service: str) -> list[str] | None:
        return train_ticket.SERVICE_CONTAINERS.get(service)

    def get_service_by_container(self, ctnr: str) -> str | None:
        return train_ticket.CONTAINER_TO_SERVICE.get(ctnr)

    def get_skip_containers(self) -> list[str]:
        return train_ticket.SKIP_CONTAINERS

    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        return train_ticket.DIAGNOSER_TARGET_DATA


def new_knowledge(target_app: str) -> PriorKnowledge:
    match target_app:
        case sock_shop.TARGET_APP_NAME:
            return SockShopKnowledge()
        case train_ticket.TARGET_APP_NAME:
            return TrainTicketKnowledge()
        case _:
            raise ValueError(f"{target_app} is invalid")
