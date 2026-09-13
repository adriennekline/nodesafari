"""Node, edge, and repeated network-robustness perturbations."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def _efficiency(graph: nx.Graph) -> float:
    return float(nx.global_efficiency(graph.to_undirected())) if graph.number_of_nodes() > 1 else 0.0


def _largest_fraction(graph: nx.Graph, denominator: int) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(graph.to_undirected())), default=0)
    return largest / max(denominator, 1)


def edge_perturbation_screen(graph: nx.Graph) -> pd.DataFrame:
    """Rank single-edge removals by efficiency loss and fragmentation."""

    undirected = nx.Graph(graph.to_undirected())
    base_efficiency = _efficiency(undirected)
    base_largest = _largest_fraction(undirected, len(undirected))
    bridges = {frozenset(edge) for edge in nx.bridges(undirected)}
    rows = []
    for source, target in undirected.edges():
        reduced = undirected.copy()
        reduced.remove_edge(source, target)
        efficiency = _efficiency(reduced)
        largest = _largest_fraction(reduced, len(undirected))
        efficiency_loss = max(base_efficiency - efficiency, 0.0)
        component_loss = max(base_largest - largest, 0.0)
        rows.append(
            {
                "source": source,
                "target": target,
                "impact_score": 0.6 * efficiency_loss + 0.4 * component_loss,
                "efficiency_change": efficiency - base_efficiency,
                "largest_component_change": largest - base_largest,
                "is_bridge": frozenset((source, target)) in bridges,
                "components_after": nx.number_connected_components(reduced),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["impact_score", "is_bridge", "source", "target"], ascending=[False, False, True, True]
    ).reset_index(drop=True)


def _targeted_order(graph: nx.Graph, count: int) -> list[object]:
    working = graph.copy()
    order = []
    for _ in range(min(count, len(working))):
        node = max(working.degree(), key=lambda item: (item[1], str(item[0])))[0]
        order.append(node)
        working.remove_node(node)
    return order


def robustness_curve(
    graph: nx.Graph,
    *,
    steps: int = 10,
    random_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare adaptive degree-targeted removal with random node failure."""

    undirected = nx.Graph(graph.to_undirected())
    n = len(undirected)
    if n == 0:
        return pd.DataFrame()
    removal_counts = sorted(set(np.linspace(0, max(n - 1, 0), steps + 1).round().astype(int)))
    targeted_order = _targeted_order(undirected, max(removal_counts, default=0))
    rng = np.random.default_rng(seed)
    rows = []

    for count in removal_counts:
        targeted = undirected.copy()
        targeted.remove_nodes_from(targeted_order[:count])
        rows.append(
            {
                "strategy": "targeted hubs",
                "fraction_removed": count / n,
                "largest_component_mean": _largest_fraction(targeted, n),
                "largest_component_std": 0.0,
                "efficiency_mean": _efficiency(targeted),
                "efficiency_std": 0.0,
            }
        )

        largest_samples = []
        efficiency_samples = []
        nodes = list(undirected)
        for _ in range(max(random_repeats, 1)):
            removed = list(rng.choice(nodes, size=count, replace=False)) if count else []
            random_graph = undirected.copy()
            random_graph.remove_nodes_from(removed)
            largest_samples.append(_largest_fraction(random_graph, n))
            efficiency_samples.append(_efficiency(random_graph))
        rows.append(
            {
                "strategy": "random failure",
                "fraction_removed": count / n,
                "largest_component_mean": float(np.mean(largest_samples)),
                "largest_component_std": float(np.std(largest_samples)),
                "efficiency_mean": float(np.mean(efficiency_samples)),
                "efficiency_std": float(np.std(efficiency_samples)),
            }
        )
    return pd.DataFrame(rows)
