import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scipy.stats
from pandas.core.groupby.generic import DataFrameGroupBy

from meltria.loader import DatasetRecord


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


def plot_rank_dataframe(
    rank_df: DataFrameGroupBy,
    fig_width: float = 20,
    graph_height: float = 2.7,
    ncols: int = 3,
) -> None:
    nrows = math.ceil(rank_df.shape[1] / ncols)
    fig, axs = plt.subplots(figsize=(fig_width, graph_height * nrows), nrows=nrows, ncols=ncols)
    for row, ax in zip(rank_df.itertuples(), axs.flatten()):  # type: ignore
        ax.plot(scipy.stats.zscore(row.metric_values))
        ax.set_title(f"{row.Index} - {row.chaos_type}/{row.chaos_comp}/{row.chaos_idx}: {row.metric_name}")
    plt.show()


def plot_dataset_dataframe(
    data_df: pd.DataFrame,
    record: DatasetRecord,
    fig_width: float = 20,
    graph_height: float = 2.7,
    ncols: int = 3,
) -> None:
    nrows = math.ceil(data_df.shape[1] / ncols)
    fig, axs = plt.subplots(figsize=(fig_width, graph_height * nrows), nrows=nrows, ncols=ncols)
    for (label, data), ax in zip(data_df.items(), axs.flatten()):  # type: ignore
        ax.plot(data.to_numpy())
        ax.set_title(f"{record.chaos_case_full()}: {label}")
    plt.show()
