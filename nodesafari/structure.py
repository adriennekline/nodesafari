"""Structural analyses beyond centrality and community detection."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .analysis import community_table, network_summary, node_metrics


def core_periphery_table(graph: nx.Graph) -> pd.DataFrame:
    """Characterize k-core position for every node.

    This is a transparent k-core decomposition, not a fitted continuous
    core-periphery block model.
    """

    undirected = nx.Graph(graph.to_undirected())
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    if not undirected:
        return pd.DataFrame(columns=["node", "core_number", "core_score", "role", "degree"])
    core = nx.core_number(undirected)
    max_core = max(core.values(), default=0)
    rows = []
    for node in sorted(undirected, key=str):
        score = core[node] / max(max_core, 1)
        if core[node] == max_core:
            role = "maximal core"
        elif score >= 0.5:
            role = "inner shell"
        else:
            role = "periphery"
        rows.append(
            {
                "node": node,
                "core_number": core[node],
                "core_score": score,
                "role": role,
                "degree": undirected.degree(node),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["core_number", "degree", "node"], ascending=[False, False, True]
    ).reset_index(drop=True)


def bridge_analysis(graph: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify articulation nodes and bridge edges with removal consequences."""

    undirected = nx.Graph(graph.to_undirected())
    articulation = set(nx.articulation_points(undirected))
    bridges = set(nx.bridges(undirected))
    betweenness = nx.betweenness_centrality(undirected, normalized=True)
    node_rows = []
    for node in sorted(undirected, key=str):
        if node not in articulation:
            continue
        reduced = undirected.copy()
        reduced.remove_node(node)
        components = nx.number_connected_components(reduced) if reduced else 0
        largest = max((len(c) for c in nx.connected_components(reduced)), default=0)
        node_rows.append(
            {
                "node": node,
                "betweenness": betweenness[node],
                "components_after_removal": components,
                "largest_component_after": largest,
            }
        )

    edge_betweenness = nx.edge_betweenness_centrality(undirected, normalized=True)
    edge_rows = []
    for source, target in sorted(bridges, key=lambda edge: (str(edge[0]), str(edge[1]))):
        reduced = undirected.copy()
        reduced.remove_edge(source, target)
        edge_rows.append(
            {
                "source": source,
                "target": target,
                "edge_betweenness": edge_betweenness.get((source, target), edge_betweenness.get((target, source), 0.0)),
                "components_after_removal": nx.number_connected_components(reduced),
            }
        )
    nodes = pd.DataFrame(node_rows, columns=["node", "betweenness", "components_after_removal", "largest_component_after"])
    edges = pd.DataFrame(edge_rows, columns=["source", "target", "edge_betweenness", "components_after_removal"])
    if not nodes.empty:
        nodes = nodes.sort_values(["components_after_removal", "betweenness"], ascending=False).reset_index(drop=True)
    if not edges.empty:
        edges = edges.sort_values(["components_after_removal", "edge_betweenness"], ascending=False).reset_index(drop=True)
    return nodes, edges


def network_statistics_table(graph: nx.Graph) -> pd.DataFrame:
    """Return global network statistics with compact interpretations."""

    undirected = graph.to_undirected()
    summary = network_summary(undirected)
    degrees = np.array([degree for _, degree in undirected.degree()], dtype=float)
    strengths = np.array([degree for _, degree in undirected.degree(weight="weight")], dtype=float)
    components = list(nx.connected_components(undirected)) if undirected else []
    largest_nodes = max(components, key=len) if components else set()
    largest = undirected.subgraph(largest_nodes)
    communities = community_table(undirected)
    partition = [set(group["node"]) for _, group in communities.groupby("community")] if not communities.empty else []
    modularity = nx.community.modularity(undirected, partition, weight="weight") if undirected.number_of_edges() and partition else 0.0
    additional = {
        "average_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "maximum_degree": float(degrees.max()) if len(degrees) else 0.0,
        "degree_std": float(degrees.std()) if len(degrees) else 0.0,
        "average_strength": float(strengths.mean()) if len(strengths) else 0.0,
        "global_efficiency": float(nx.global_efficiency(undirected)) if len(undirected) > 1 else 0.0,
        "modularity": float(modularity),
        "largest_component_diameter": float(nx.diameter(largest)) if len(largest) > 1 else 0.0,
        "largest_component_mean_path": float(nx.average_shortest_path_length(largest)) if len(largest) > 1 else 0.0,
    }
    interpretations = {
        "nodes": "Number of entities in the network.",
        "edges": "Number of observed interactions.",
        "density": "Fraction of possible interactions that are present.",
        "components": "Number of disconnected network regions.",
        "largest_component_fraction": "Fraction of nodes in the largest connected region.",
        "average_clustering": "Mean weighted tendency for neighbors to connect.",
        "transitivity": "Global fraction of connected triples that form triangles.",
        "degree_assortativity": "Preference for nodes to connect to similarly connected nodes.",
        "average_degree": "Mean number of interactions per node.",
        "maximum_degree": "Largest interaction count for any node.",
        "degree_std": "Variation in interaction counts.",
        "average_strength": "Mean sum of edge weights per node.",
        "global_efficiency": "Average inverse shortest-path distance.",
        "modularity": "Separation strength of detected communities.",
        "largest_component_diameter": "Longest shortest path in the largest component.",
        "largest_component_mean_path": "Mean shortest path within the largest component.",
    }
    values = {**summary, **additional}
    return pd.DataFrame(
        [{"metric": key, "value": value, "interpretation": interpretations[key]} for key, value in values.items()]
    )


def structural_node_table(graph: nx.Graph) -> pd.DataFrame:
    """Combine centrality, community, core, and bridge roles by node."""

    table = node_metrics(graph)
    table = table.merge(community_table(graph), on="node", how="left")
    table = table.merge(core_periphery_table(graph), on=["node", "degree"], how="left")
    articulation, _ = bridge_analysis(graph)
    table["is_articulation"] = table["node"].isin(set(articulation.get("node", [])))
    return table
