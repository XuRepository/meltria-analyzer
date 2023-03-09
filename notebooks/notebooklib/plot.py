import matplotlib.pyplot as plt
import networkx as nx


def plot_causal_graph(G: nx.Graph) -> None:
    plt.figure(figsize=(20, 20))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G, pos=pos, font_size=8)
    # nx.set_edge_attributes(g, {(e[0], e[1]): {'label': e[2]['weight']} for e in g.edges(data=True)})
    labels = {k: round(v, 2) for k, v in nx.get_edge_attributes(G, "weight").items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=6)
    plt.show()
