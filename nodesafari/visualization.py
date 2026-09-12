"""Matplotlib-based visual helpers shared by the Streamlit interface."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PALETTE = ["#18B6A4", "#6C63FF", "#F59E5B", "#EE6C91", "#55A8E8", "#A8C95F"]


def network_figure(graph: nx.Graph, communities: pd.DataFrame, seed: int = 42):
    """Draw a readable, reproducible force-directed network view."""

    fig, ax = plt.subplots(figsize=(9, 6.4), facecolor="#07121F")
    ax.set_facecolor("#07121F")
    positions = nx.spring_layout(graph, seed=seed, weight="weight", k=1 / max(np.sqrt(graph.number_of_nodes()), 1))
    assignments = dict(zip(communities.get("node", []), communities.get("community", [])))
    colors = [PALETTE[(int(assignments.get(node, 1)) - 1) % len(PALETTE)] for node in graph]
    sizes = [90 + 36 * graph.degree(node) for node in graph]
    widths = [0.35 + 0.45 * min(float(data.get("weight", 1.0)), 4.0) for _, _, data in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#6E829A", width=widths, alpha=0.28)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors, node_size=sizes, linewidths=1.2, edgecolors="#D8F5F0")
    if graph.number_of_nodes() <= 35:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=8, font_color="#EAF3F8")
    ax.margins(0.10)
    ax.axis("off")
    fig.tight_layout()
    return fig
