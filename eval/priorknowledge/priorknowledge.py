from collections import defaultdict

import networkx as nx
from eval.priorknowledge import sock_shop, train_ticket


class PriorKnowledge:
    def __init__(self, target_app: str) -> None:
        self.target_app = target_app

    def get_root_service(self):
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.ROOT_SERVICE
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.ROOT_SERVICE
            case _:
                raise ValueError(f'{self.target_app} is invalid')

    def get_containers(self, skip: bool = False) -> list[str]:
        ctnrs: list[str]
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                ctnrs = list(sock_shop.CONTAINER_CALL_GRAPH.keys())
            case train_ticket.TARGET_APP_NAME:
                ctnrs = list(train_ticket.CONTAINER_CALL_GRAPH.keys())
            case _:
                raise ValueError(f'{self.target_app} is invalid')
        if skip:
            return [ctnr for ctnr in ctnrs if ctnr not in self.get_skip_containers()]
        return ctnrs

    def get_root_metrics(self) -> tuple[str, ...]:
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.ROOT_METRIC_LABELS
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.ROOT_METRIC_LABELS
            case _:
                raise ValueError(f'{self.target_app} is invalid')

    def get_service_call_digraph(self) -> nx.DiGraph:
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.SERVICE_CALL_DIGRAPH
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.SERVICE_CALL_DIGRAPH
            case _:
                raise ValueError(f'{self.target_app} is invalid')

    def get_container_call_digraph(self) -> nx.DiGraph:
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.CONTAINER_CALL_DIGRAPH
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.CONTAINER_CALL_DIGRAPH
            case _:
                raise ValueError(f'{self.target_app} is invalid')

    def get_container_call_graph(self, ctnr: str) -> list[str]:
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.CONTAINER_CALL_GRAPH[ctnr]
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.CONTAINER_CALL_GRAPH[ctnr]
            case _:
                raise ValueError(f'{self.target_app} is invalid')

    def get_service_routes(self, service: str) -> list[tuple[str, ...]]:
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.SERVICE_TO_SERVICE_ROUTES[service]
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.SERVICE_TO_SERVICE_ROUTES[service]
            case _:
                raise ValueError(f'{self.target_app} is invalid')

    def get_service_containers(self, service: str) -> list[str]:
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.SERVICE_CONTAINERS[service]
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.SERVICE_CONTAINERS[service]
            case _:
                raise ValueError(f'{self.target_app} is invalid')

    def get_service_by_container(self, ctnr: str) -> str:
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.CONTAINER_TO_SERVICE[ctnr]
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.CONTAINER_TO_SERVICE[ctnr]
            case _:
                raise ValueError(f'{self.target_app} is invalid')

    def get_skip_containers(self) -> list[str]:
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.SKIP_CONTAINERS
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.SKIP_CONTAINERS
            case _:
                raise ValueError(f'{self.target_app} is invalid')

    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        match self.target_app:
            case sock_shop.TARGET_APP_NAME:
                return sock_shop.DIAGNOSER_TARGET_DATA
            case train_ticket.TARGET_APP_NAME:
                return train_ticket.DIAGNOSER_TARGET_DATA
            case _:
                raise ValueError(f'{self.target_app} is invalid')

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
