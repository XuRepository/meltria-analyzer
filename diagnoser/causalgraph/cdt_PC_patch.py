import networkx as nx
from cdt.causality.graph import PC
from cdt.causality.graph.model import GraphModel
from pandas import DataFrame


def register_cls_method(cls, method_name: str, method) -> None:
    setattr(cls, method_name, lambda *args, **kwargs: method(*args, **kwargs))


def create_graph_from_init_graph_as_interface(self, data, init_graph, **kwargs):
    raise NotImplementedError


def create_graph_from_init_graph(self, data, init_graph, **kwargs):
    # Building setup w/ arguments.
    self.arguments["{CITEST}"] = self.dir_CI_test[self.CI_test]
    self.arguments["{METHOD_INDEP}"] = self.dir_method_indep[self.CI_test]
    self.arguments["{DIRECTED}"] = "TRUE"
    self.arguments["{ALPHA}"] = str(self.alpha)
    self.arguments["{NJOBS}"] = str(self.njobs)
    self.arguments["{VERBOSE}"] = str(self.verbose).upper()

    graph = nx.complement(init_graph)
    fg = DataFrame(nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()), weight=None).todense())

    results = self._run_pc(data, fixedEdges=None, fixedGaps=fg, verbose=self.verbose)

    return nx.relabel_nodes(nx.DiGraph(results), {idx: i for idx, i in enumerate(sorted(data.columns))})


register_cls_method(GraphModel, "create_graph_from_init_graph", create_graph_from_init_graph_as_interface)
register_cls_method(PC, "create_graph_from_init_graph", create_graph_from_init_graph)
