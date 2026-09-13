"""Matplotlib-based visual helpers shared by the Streamlit interface."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .analysis import Richness, rich_club_edge_roles, rich_club_members

PALETTE = ["#2DD4BF", "#8B5CF6", "#14B8A6", "#A78BFA", "#5EEAD4", "#7C3AED"]


def rich_club_curve_figure(curve: pd.DataFrame, *, richness: str = "degree"):
    """Create a publication-oriented observed/null and normalized rich-club figure."""

    x = curve["threshold"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    axes[0].fill_between(
        x,
        curve["phi_null_lower_95"].to_numpy(dtype=float),
        curve["phi_null_upper_95"].to_numpy(dtype=float),
        color="#c4b5fd",
        alpha=0.35,
        label="95% null envelope",
    )
    axes[0].plot(x, curve["phi_null_mean"], color="#7c3aed", label="Null mean")
    axes[0].plot(x, curve["phi_observed"], color="#0f9f91", linewidth=2.2, label="Observed")
    axes[0].set_ylabel("Rich-club coefficient φ")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(1.0, color="#7c3aed", linestyle="--", linewidth=1)
    axes[1].plot(x, curve["rho"], color="#0f9f91", linewidth=2.2)
    signal = curve["exploratory_signal"].to_numpy(dtype=bool)
    unreliable = ~curve["reliable_node_count"].to_numpy(dtype=bool)
    axes[1].scatter(x[signal], curve.loc[signal, "rho"], color="#f97316", label="Signal")
    axes[1].scatter(
        x[unreliable],
        curve.loc[unreliable, "rho"],
        facecolors="none",
        edgecolors="#9ca3af",
        label="Small rich set",
    )
    axes[1].set_ylabel("Normalized coefficient ρ")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xlabel(f"{richness.capitalize()} threshold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    return fig


def network_figure(graph: nx.Graph, communities: pd.DataFrame, seed: int = 42):
    """Draw a readable, reproducible force-directed network view."""

    fig, ax = plt.subplots(figsize=(9, 6.4), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    positions = nx.spring_layout(
        graph, seed=seed, weight="weight", k=1 / max(np.sqrt(graph.number_of_nodes()), 1)
    )
    assignments = dict(zip(communities.get("node", []), communities.get("community", [])))
    colors = [PALETTE[(int(assignments.get(node, 1)) - 1) % len(PALETTE)] for node in graph]
    sizes = [90 + 36 * graph.degree(node) for node in graph]
    widths = [
        0.35 + 0.45 * min(float(data.get("weight", 1.0)), 4.0)
        for _, _, data in graph.edges(data=True)
    ]
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#968DAA", width=widths, alpha=0.4)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=colors,
        node_size=sizes,
        linewidths=1.5,
        edgecolors="#FFFFFF",
    )
    if graph.number_of_nodes() <= 35:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=8, font_color="#FFFFFF")
    ax.margins(0.10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def rich_club_network_figure(
    graph: nx.Graph,
    threshold: float,
    *,
    richness: Richness = "degree",
    seed: int = 42,
):
    """Draw rich-club membership and rich/feeder/local edge roles."""

    undirected = nx.Graph(graph.to_undirected())
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    positions = nx.spring_layout(undirected, seed=seed, weight="weight")
    members = rich_club_members(undirected, threshold, richness=richness)
    roles = rich_club_edge_roles(undirected, threshold, richness=richness)
    role_lookup = {
        frozenset((row.source, row.target)): row.edge_class for row in roles.itertuples()
    }

    fig, ax = plt.subplots(figsize=(9, 6.2), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    edge_styles = {
        "local": ("#cbd5e1", 1.0, 0.65),
        "feeder": ("#8b5cf6", 1.7, 0.8),
        "rich-club": ("#0f9f91", 2.6, 0.95),
    }
    for edge_class, (color, width, alpha) in edge_styles.items():
        edges = [edge for edge in undirected.edges() if role_lookup[frozenset(edge)] == edge_class]
        nx.draw_networkx_edges(
            undirected,
            positions,
            edgelist=edges,
            ax=ax,
            edge_color=color,
            width=width,
            alpha=alpha,
        )

    non_members = [node for node in undirected if node not in members]
    rich_members = [node for node in undirected if node in members]
    nx.draw_networkx_nodes(
        undirected,
        positions,
        nodelist=non_members,
        ax=ax,
        node_color="#ede9fe",
        edgecolors="#8b5cf6",
        node_size=[90 + 32 * undirected.degree(node) for node in non_members],
        linewidths=1.0,
    )
    nx.draw_networkx_nodes(
        undirected,
        positions,
        nodelist=rich_members,
        ax=ax,
        node_color="#14b8a6",
        edgecolors="#0f766e",
        node_size=[110 + 38 * undirected.degree(node) for node in rich_members],
        linewidths=1.3,
    )
    if undirected.number_of_nodes() <= 35:
        nx.draw_networkx_labels(undirected, positions, ax=ax, font_size=8, font_color="#201a3b")

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#14b8a6",
            markeredgecolor="#0f766e",
            label="Rich member",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#ede9fe",
            markeredgecolor="#8b5cf6",
            label="Non-member",
        ),
        Line2D([0], [0], color="#0f9f91", lw=2.6, label="Rich-club edge"),
        Line2D([0], [0], color="#8b5cf6", lw=1.7, label="Feeder edge"),
        Line2D([0], [0], color="#cbd5e1", lw=1.2, label="Local edge"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=8)
    ax.set_title(f"Rich-club roles at {richness} threshold > {threshold:g}", color="#30294e")
    ax.margins(0.1)
    ax.axis("off")
    fig.tight_layout()
    return fig
