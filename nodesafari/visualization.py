"""Matplotlib-based visual helpers shared by the Streamlit interface."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PALETTE = ["#2DD4BF", "#8B5CF6", "#14B8A6", "#A78BFA", "#5EEAD4", "#7C3AED"]


def network_figure(graph: nx.Graph, communities: pd.DataFrame, seed: int = 42):
    """Draw a readable, reproducible force-directed network view."""

    fig, ax = plt.subplots(figsize=(9, 6.4), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    positions = nx.spring_layout(graph, seed=seed, weight="weight", k=1 / max(np.sqrt(graph.number_of_nodes()), 1))
    assignments = dict(zip(communities.get("node", []), communities.get("community", [])))
    colors = [PALETTE[(int(assignments.get(node, 1)) - 1) % len(PALETTE)] for node in graph]
    sizes = [90 + 36 * graph.degree(node) for node in graph]
    widths = [0.35 + 0.45 * min(float(data.get("weight", 1.0)), 4.0) for _, _, data in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#968DAA", width=widths, alpha=0.4)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors, node_size=sizes, linewidths=1.5, edgecolors="#FFFFFF")
    if graph.number_of_nodes() <= 35:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=8, font_color="#FFFFFF")
    ax.margins(0.10)
    ax.axis("off")
    fig.tight_layout()
    return fig
