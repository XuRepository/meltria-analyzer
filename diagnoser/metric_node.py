from enum import IntEnum, auto, unique
from functools import total_ordering

import networkx as nx
import pandas as pd


@unique
class MetricType(IntEnum):
    CONTAINER = auto()
    SERVICE = auto()
    NODE = auto()
    MIDDLEWARE = auto()


@total_ordering
class MetricNode(str):
    # label_ should be like 'c-orders_cpu_usage_seconds_total'
    def __new__(cls, label_: str):
        self = super().__new__(cls, label_)
        return self

    def __init__(self, label_: str) -> None:
        self.label = label_
        self.comp, self.base_name = label_.split("-", maxsplit=1)[1].split("_", maxsplit=1)
        if self.startswith("c-"):
            self.comp_type = MetricType.CONTAINER
        elif self.startswith("s-"):
            self.comp_type = MetricType.SERVICE
        elif self.startswith("n-"):
            self.comp_type = MetricType.NODE
        elif self.startswith("m-"):
            self.comp_type = MetricType.MIDDLEWARE
        else:
            raise ValueError(f"no prefix: {label_}")

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.label == other.label

    def __lt__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.label < other.label

    def __hash__(self) -> int:
        return hash(self.label)

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return self.label

    def is_service(self) -> bool:
        return self.comp_type == MetricType.SERVICE

    def is_container(self) -> bool:
        return self.comp_type == MetricType.CONTAINER

    def is_node(self) -> bool:
        return self.comp_type == MetricType.NODE

    def is_middleware(self) -> bool:
        return self.comp_type == MetricType.MIDDLEWARE

    def is_container_or_middleware(self) -> bool:
        return self.is_container() or self.is_middleware()


class MetricNodes(object):
    def __init__(self, num_to_node: dict[int, MetricNode]) -> None:
        self.nodes: list[MetricNode] = []
        self._i = 0
        self.node_to_num: dict[MetricNode, int] = {}
        self.num_to_node: dict[int, MetricNode] = num_to_node
        for i, n in num_to_node.items():
            self.nodes.append(n)
            self.node_to_num[n] = i

    @classmethod
    def from_num_to_label(cls, num_to_label: dict[int, str]):
        return cls({i: MetricNode(v) for i, v in num_to_label.items()})

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return cls({i: MetricNode(v) for i, v in enumerate(df.columns)})

    @classmethod
    def from_list_of_metric_node(cls, nodelist: list[MetricNode]):
        return cls({i: v for i, v in enumerate(nodelist)})

    @classmethod
    def from_metric_names(cls, metric_names: list[str]):
        return cls({i: MetricNode(v) for i, v in enumerate(metric_names)})

    def __iter__(self):
        yield from self.nodes

    def __next__(self):
        if self._i >= len(self.nodes):
            raise StopIteration()
        ret = self.nodes[self._i]
        self._i += 1
        return ret

    def __str__(self) -> str:
        return ",".join([n.label for n in self.nodes])

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.nodes)

    def liststr(self) -> str:
        return "[" + ",".join([n.label for n in self.nodes]) + "]"

    def tolist(self) -> list[str]:
        return [n.label for n in self.nodes]

    def node_to_label(self) -> dict[MetricNode, str]:
        return {n: n.label for n in self.nodes}

    def label_to_node(self) -> dict[str, MetricNode]:
        return {n.label: n for n in self.nodes}

    def num_to_node_for_causallearn(self) -> dict[str, MetricNode]:
        # causallearn set node name as X0, X1, ...
        return {f"X{i+1}": node for i, node in self.num_to_node.items()}


def relabel_graph_nodes_to_label(G: nx.Graph) -> nx.Graph:
    mapping = MetricNodes.from_list_of_metric_node(list(G.nodes)).node_to_label()
    return nx.relabel_nodes(G, mapping, copy=True)


def relabel_graph_labels_to_node(G: nx.Graph) -> nx.Graph:
    mapping = MetricNodes.from_metric_names(list(G.nodes)).label_to_node()
    return nx.relabel_nodes(G, mapping, copy=True)
