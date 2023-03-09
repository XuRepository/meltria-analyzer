import matplotlib.pyplot as plt
import networkx as nx


def plot_causal_graph(
    G: nx.Graph, figsize: tuple[float, float] = (20, 20), layout_scale: int = 1, node_label_y_off: float = 0.01
) -> None:
    plt.figure(figsize=figsize)

    pos = nx.kamada_kawai_layout(G, scale=layout_scale)
    nx.draw_networkx(G, pos=pos, with_labels=False)

    node_labels_pos = {k: (v[0], v[1] - node_label_y_off) for k, v in pos.items()}
    nx.draw_networkx_labels(G, pos=node_labels_pos, font_size=8)

    edge_labels = {k: round(w, 2) for k, w in nx.get_edge_attributes(G, "weight").items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, label_pos=0.25)

    plt.show()
