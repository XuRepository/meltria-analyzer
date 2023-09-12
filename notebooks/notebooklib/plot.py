import math
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from pandas.core.groupby.generic import DataFrameGroupBy
from sklearn.preprocessing import minmax_scale

from eval.groundtruth import check_cause_metrics
from meltria.loader import DatasetRecord
from tsdr import tsdr


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
    grid: bool = False,
    logging: bool = False,
) -> None:
    nrows = math.ceil(data_df.shape[1] / ncols)
    fig, axs = plt.subplots(figsize=(fig_width, graph_height * nrows), nrows=nrows, ncols=ncols)
    for (label, data), ax in zip(data_df.items(), axs.flatten()):  # type: ignore
        ax.plot(data.to_numpy(), label=label)
        ax.set_title(f"{record.chaos_case_full()}: {label}")
        if logging:
            print(f"{record.chaos_case_full()}: {label}", file=sys.stderr)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(int(math.ceil(start)), int(math.ceil(end)), 1))
        ax.grid(grid)
    plt.show()


def plot_sli_and_causal_metrics(
    data_df: pd.DataFrame,
    record: DatasetRecord,
    optional_cause: bool = True,
    fig_width: float = 20,
    graph_height: float = 2.7,
    ncols: int = 3,
    stacked: bool = False,
    n_metrics_per_graph: int = 3,
    grid: bool = False,
    logging: bool = False,
    change_points: dict[str, int] = {},
    title_label_only: bool = False,
):
    root_metrics = [m for m in list(record.pk.get_root_metrics()) if m in data_df.columns.tolist()]
    ok, cause_metrics = check_cause_metrics(
        record.pk,
        data_df.columns.tolist(),
        record.chaos_type(),
        record.chaos_comp(),
        optional_cause=optional_cause,
    )
    assert ok, "The causal metrics are not correct."

    data_df = pd.concat([record.data_df.loc[:, root_metrics], data_df.loc[:, cause_metrics.tolist()]], axis=1)
    if stacked:
        nrows = math.ceil(data_df.shape[1] / (ncols * n_metrics_per_graph))
        fig, axs = plt.subplots(figsize=(fig_width, graph_height * nrows), nrows=nrows, ncols=ncols)
        args = [iter(data_df.items())] * n_metrics_per_graph
        for items, ax in zip(zip(*args), axs.flatten()):  # type: ignore
            for label, data in items:
                ax.plot(minmax_scale(data.to_numpy()), label=label)
            ax.legend(loc="upper left")
            ax.set_title(f"{record.chaos_case_full()}")
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(int(math.ceil(start)), int(math.ceil(end)), 1))
            ax.grid(grid)
        fig.tight_layout()
        return fig
    else:
        nrows = math.ceil(data_df.shape[1] / ncols)
        fig, axs = plt.subplots(figsize=(fig_width, graph_height * nrows), nrows=nrows, ncols=ncols)
        for (label, data), ax in zip(data_df.items(), axs.flatten()):  # type: ignore
            y = data.to_numpy()
            ax.plot(y, label=label, color="black", linewidth=2)
            if (cp := change_points.get(label, {})):
                ax.plot(cp, y[cp], "X", color="red", markersize=16)
            if title_label_only:
                ax.set_title(label, fontsize=16)
            else:
                ax.set_title(f"{record.chaos_case_full()}: {label}", fontsize=16)
            if logging:
                print(f"{record.chaos_case_full()}: {label}", file=sys.stderr)
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(int(math.ceil(start)), int(math.ceil(end)), 2))
            ax.xaxis.set_tick_params(labelsize=14)
            ax.grid(grid)
        fig.tight_layout()
        return fig
    # plt.show()


# def plot_distribution_corr_sli_and_metrics(
#     data_df: pd.DataFrame,
#     record: DatasetRecord,
#     optional_cause: bool = True,
#     fig_width: float = 20,
#     sli_metrics: list[str] = [],
# ):
#     sli_metrics = [m for m in sli_metrics or list(record.pk.get_root_metrics()) if m in data_df.columns.tolist()]
#     ok, cause_metrics = check_cause_metrics(
#         record.pk,
#         data_df.columns.tolist(),
#         record.chaos_type(),
#         record.chaos_comp(),
#         optional_cause=optional_cause,
#     )
#     assert ok, "The causal metrics are not correct."
#     metrics = record.data_df.loc[~record.data_df.index.isin(sli_metrics)]

# for sli_metric in sli_metrics:


def plot_distribution_corr_with_cause_metrics(
    record: DatasetRecord,
    optional_cause: bool = True,
):
    data_df = tsdr.filter_out_no_change_metrics(record.data_df, parallel=True)
    ok, cause_metrics = check_cause_metrics(
        record.pk,
        data_df.columns.tolist(),
        record.chaos_type(),
        record.chaos_comp(),
        optional_cause=optional_cause,
    )
    assert ok, "The causal metrics are not correct."
    cause_metrics_df = data_df[cause_metrics]
    other_metrics_df = data_df[~record.data_df.index.isin(cause_metrics)]
    corrs: dict[str, list[float]] = defaultdict(lambda: [])
    for cause_metric in cause_metrics_df.columns:
        for other_metric in other_metrics_df.columns:
            res = scipy.stats.pearsonr(
                x=cause_metrics_df[cause_metric].to_numpy(), y=other_metrics_df[other_metric].to_numpy()
            )
            corrs[cause_metric].append(np.abs(res[0]))
    fig, ax = plt.subplots(figsize=(20, 6))
    sns.boxplot(data=pd.DataFrame.from_dict(corrs), ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()
