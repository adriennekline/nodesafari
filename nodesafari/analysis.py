"""Interpretable network-science and graph-ML analyses."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def _safe_assortativity(graph: nx.Graph) -> float:
    if graph.number_of_edges() < 2:
        return float("nan")
    with np.errstate(all="ignore"):
        value = nx.degree_assortativity_coefficient(graph)
    return float(value) if np.isfinite(value) else float("nan")


def network_summary(graph: nx.Graph) -> dict[str, float | int]:
    """Return a compact collection of whole-network descriptors."""

    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected)) if graph else []
    largest = max((len(c) for c in components), default=0)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "components": len(components),
        "largest_component_fraction": largest / max(graph.number_of_nodes(), 1),
        "average_clustering": nx.average_clustering(undirected, weight="weight")
        if graph.number_of_nodes() > 1
        else 0.0,
        "transitivity": nx.transitivity(undirected),
        "degree_assortativity": _safe_assortativity(undirected),
    }


def node_metrics(graph: nx.Graph) -> pd.DataFrame:
    """Calculate complementary centrality measures for each node."""

    if graph.number_of_nodes() == 0:
        return pd.DataFrame()
    undirected = graph.to_undirected()
    degree = nx.degree_centrality(undirected)
    strength = dict(undirected.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(undirected, weight=None, normalized=True)
    closeness = nx.closeness_centrality(undirected)
    pagerank = nx.pagerank(undirected, weight="weight")
    try:
        eigenvector = nx.eigenvector_centrality_numpy(undirected, weight="weight")
    except (nx.NetworkXException, TypeError):
        eigenvector = {node: float("nan") for node in undirected}

    rows = [
        {
            "node": node,
            "degree": undirected.degree(node),
            "strength": strength[node],
            "degree_centrality": degree[node],
            "betweenness": betweenness[node],
            "closeness": closeness[node],
            "pagerank": pagerank[node],
            "eigenvector": eigenvector[node],
        }
        for node in undirected
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["degree_centrality", "betweenness"], ascending=False)
        .reset_index(drop=True)
    )


def _community_partition(graph: nx.Graph) -> list[set[str]]:
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [{str(node)} for node in graph]
    communities = nx.community.greedy_modularity_communities(graph.to_undirected(), weight="weight")
    return [set(group) for group in communities]


def community_table(graph: nx.Graph) -> pd.DataFrame:
    """Assign nodes to deterministic greedy-modularity communities."""

    records: list[dict[str, object]] = []
    for community_id, members in enumerate(_community_partition(graph), start=1):
        for node in sorted(members, key=str):
            records.append({"node": node, "community": community_id})
    return pd.DataFrame(records)


Richness = Literal["degree", "strength"]


def _richness_values(graph: nx.Graph, richness: Richness) -> dict[object, float]:
    if richness == "degree":
        return {node: float(value) for node, value in graph.degree()}
    if richness == "strength":
        return {node: float(value) for node, value in graph.degree(weight="weight")}
    raise ValueError("richness must be 'degree' or 'strength'.")


def _rich_club_phi(
    graph: nx.Graph,
    thresholds: Iterable[float],
    *,
    richness: Richness = "degree",
    weighted: bool = False,
) -> tuple[dict[float, float], dict[float, int], dict[float, int]]:
    scores = _richness_values(graph, richness)
    strongest_weights = sorted(
        (float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True)),
        reverse=True,
    )
    values: dict[float, float] = {}
    node_counts: dict[float, int] = {}
    edge_counts: dict[float, int] = {}
    for threshold in thresholds:
        members = [node for node, score in scores.items() if score > threshold]
        count = len(members)
        subgraph = graph.subgraph(members)
        edge_count = subgraph.number_of_edges()
        node_counts[threshold] = count
        edge_counts[threshold] = edge_count
        if count < 2:
            values[threshold] = float("nan")
            continue
        if weighted:
            numerator = sum(
                float(data.get("weight", 1.0)) for _, _, data in subgraph.edges(data=True)
            )
            denominator = sum(strongest_weights[:edge_count])
            values[threshold] = numerator / denominator if denominator > 0 else float("nan")
        else:
            values[threshold] = 2 * edge_count / (count * (count - 1))
    return values, node_counts, edge_counts


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    ranked = p_values[order] * finite.size / np.arange(1, finite.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def rich_club_members(
    graph: nx.Graph, threshold: float, *, richness: Richness = "degree"
) -> set[object]:
    """Return nodes whose selected richness is strictly greater than a threshold."""

    undirected = nx.Graph(graph.to_undirected())
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    scores = _richness_values(undirected, richness)
    return {node for node, score in scores.items() if score > threshold}


def rich_club_edge_roles(
    graph: nx.Graph, threshold: float, *, richness: Richness = "degree"
) -> pd.DataFrame:
    """Classify edges as rich-club, feeder, or local at one threshold."""

    undirected = nx.Graph(graph.to_undirected())
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    members = rich_club_members(undirected, threshold, richness=richness)
    records = []
    for source, target, data in undirected.edges(data=True):
        source_rich = source in members
        target_rich = target in members
        edge_class = (
            "rich-club"
            if source_rich and target_rich
            else "feeder"
            if source_rich or target_rich
            else "local"
        )
        records.append(
            {
                "source": source,
                "target": target,
                "weight": float(data.get("weight", 1.0)),
                "edge_class": edge_class,
            }
        )
    return pd.DataFrame(records)


def rich_club_curve(
    graph: nx.Graph,
    *,
    randomizations: int = 20,
    seed: int = 42,
    swaps_per_edge: int = 10,
    min_rich_nodes: int = 5,
    richness: Richness = "degree",
    weighted: bool = False,
) -> pd.DataFrame:
    """Estimate observed and degree-preserving normalized rich-club curves.

    Normalized values above one indicate denser-than-null connectivity among
    nodes whose degree exceeds the corresponding threshold. The null ensemble
    uses degree-preserving double-edge swaps. Weighted analysis uses an
    Opsahl-style coefficient and shuffles the observed weights over rewired
    edges; it preserves the global weight distribution but not node strength.
    """

    if randomizations < 1:
        raise ValueError("randomizations must be at least 1.")
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge cannot be negative.")
    if min_rich_nodes < 2:
        raise ValueError("min_rich_nodes must be at least 2.")
    if richness == "strength" and not weighted:
        raise ValueError("Strength-based richness requires weighted=True.")

    undirected = nx.Graph(graph.to_undirected())
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    if undirected.number_of_nodes() < 3 or undirected.number_of_edges() == 0:
        raise ValueError("Rich-club analysis requires at least three nodes and one edge.")
    if weighted:
        weights = np.asarray(
            [float(data.get("weight", 1.0)) for _, _, data in undirected.edges(data=True)]
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Weighted rich-club analysis requires finite, non-negative weights.")

    scores = np.asarray(list(_richness_values(undirected, richness).values()), dtype=float)
    if richness == "degree":
        thresholds = [float(value) for value in range(int(scores.max(initial=0)))]
    else:
        thresholds = np.unique(scores)[:-1].astype(float).tolist()
    observed, node_counts, edge_counts = _rich_club_phi(
        undirected, thresholds, richness=richness, weighted=weighted
    )
    null_matrix = np.full((randomizations, len(thresholds)), np.nan, dtype=float)

    rng = np.random.default_rng(seed)
    for randomization in range(randomizations):
        null_graph = nx.Graph()
        null_graph.add_nodes_from(undirected.nodes())
        null_graph.add_edges_from(undirected.edges())
        edges = null_graph.number_of_edges()
        swaps = int(swaps_per_edge * edges)
        if null_graph.number_of_nodes() >= 4 and edges >= 2 and swaps > 0:
            try:
                nx.double_edge_swap(
                    null_graph,
                    nswap=swaps,
                    max_tries=max(100, swaps * 20),
                    seed=int(rng.integers(0, 2**32 - 1)),
                )
            except nx.NetworkXAlgorithmError:
                pass
        if weighted:
            shuffled_weights = np.asarray(
                [float(data.get("weight", 1.0)) for _, _, data in undirected.edges(data=True)]
            )
            rng.shuffle(shuffled_weights)
            for (source, target), weight in zip(null_graph.edges(), shuffled_weights, strict=True):
                null_graph[source][target]["weight"] = float(weight)
        phi, _, _ = _rich_club_phi(null_graph, thresholds, richness=richness, weighted=weighted)
        null_matrix[randomization] = [phi[threshold] for threshold in thresholds]

    rows = []
    p_values = np.full(len(thresholds), np.nan, dtype=float)
    for index, threshold in enumerate(thresholds):
        samples = null_matrix[:, index]
        samples = samples[np.isfinite(samples)]
        null_mean = float(np.mean(samples)) if samples.size else float("nan")
        lower, upper = (
            np.quantile(samples, [0.025, 0.975]) if samples.size else (float("nan"), float("nan"))
        )
        obs = observed[threshold]
        normalized = obs / null_mean if null_mean > 0 and np.isfinite(obs) else float("nan")
        if np.isfinite(obs) and samples.size:
            p_values[index] = (1 + np.sum(samples >= obs)) / (samples.size + 1)
        rows.append(
            {
                "threshold": threshold,
                "degree_threshold": threshold,
                "n_rich_nodes": node_counts[threshold],
                "n_rich_edges": edge_counts[threshold],
                "phi_observed": obs,
                "phi_null_mean": null_mean,
                "phi_null_lower_95": float(lower),
                "phi_null_upper_95": float(upper),
                "rho": normalized,
                "observed_phi": obs,
                "null_phi": null_mean,
                "normalized_phi": normalized,
            }
        )
    result = pd.DataFrame(rows)
    result["p_empirical"] = p_values
    result["q_bh"] = _benjamini_hochberg(p_values)
    result["reliable_node_count"] = result["n_rich_nodes"] >= min_rich_nodes
    result["exploratory_signal"] = (
        (result["rho"] > 1) & (result["p_empirical"] < 0.05) & result["reliable_node_count"]
    )
    result.attrs["parameters"] = {
        "richness": richness,
        "weighted": weighted,
        "randomizations": randomizations,
        "swaps_per_edge": swaps_per_edge,
        "seed": seed,
        "min_rich_nodes": min_rich_nodes,
    }
    result.attrs["warnings"] = [
        *(
            [
                (
                    "Weighted nulls preserve degree and the global weight distribution, "
                    "but not each node's strength; treat weighted inference as exploratory."
                )
            ]
            if weighted
            else []
        ),
        *(
            ["Use at least 1,000 null networks for final inference."]
            if randomizations < 1000
            else []
        ),
    ]
    return result


def _global_efficiency(graph: nx.Graph) -> float:
    if graph.number_of_nodes() < 2:
        return 0.0
    return float(nx.global_efficiency(graph.to_undirected()))


def perturbation_screen(graph: nx.Graph) -> pd.DataFrame:
    """Rank single-node deletions by loss of connectivity and efficiency."""

    undirected = graph.to_undirected()
    base_n = max(undirected.number_of_nodes(), 1)
    base_efficiency = _global_efficiency(undirected)
    base_largest = max((len(c) for c in nx.connected_components(undirected)), default=0)

    records = []
    for node in undirected:
        perturbed = undirected.copy()
        perturbed.remove_node(node)
        largest = max((len(c) for c in nx.connected_components(perturbed)), default=0)
        efficiency = _global_efficiency(perturbed)
        component_loss = max(base_largest - largest - 1, 0) / base_n
        efficiency_loss = max(base_efficiency - efficiency, 0.0)
        impact = 0.6 * efficiency_loss + 0.4 * component_loss
        records.append(
            {
                "node": node,
                "impact_score": impact,
                "efficiency_change": efficiency - base_efficiency,
                "largest_component_change": largest - (base_largest - 1),
                "components_after": nx.number_connected_components(perturbed)
                if perturbed.number_of_nodes()
                else 0,
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(["impact_score", "node"], ascending=[False, True])
        .reset_index(drop=True)
    )


def compare_networks(graph_a: nx.Graph, graph_b: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare global structure and node-level degree/centrality changes."""

    summary_a = network_summary(graph_a)
    summary_b = network_summary(graph_b)
    global_rows = []
    for metric in summary_a:
        a = summary_a[metric]
        b = summary_b[metric]
        global_rows.append({"metric": metric, "network_a": a, "network_b": b, "change": b - a})

    all_nodes = sorted(set(graph_a) | set(graph_b), key=str)
    centrality_a = (
        nx.degree_centrality(graph_a.to_undirected()) if graph_a.number_of_nodes() > 1 else {}
    )
    centrality_b = (
        nx.degree_centrality(graph_b.to_undirected()) if graph_b.number_of_nodes() > 1 else {}
    )
    node_rows = []
    for node in all_nodes:
        degree_a = graph_a.degree(node) if node in graph_a else 0
        degree_b = graph_b.degree(node) if node in graph_b else 0
        node_rows.append(
            {
                "node": node,
                "degree_a": degree_a,
                "degree_b": degree_b,
                "degree_change": degree_b - degree_a,
                "centrality_change": centrality_b.get(node, 0.0) - centrality_a.get(node, 0.0),
                "status": "gained"
                if node not in graph_a
                else "lost"
                if node not in graph_b
                else "shared",
            }
        )
    nodes = pd.DataFrame(node_rows)
    nodes["absolute_change"] = nodes["degree_change"].abs()
    nodes = nodes.sort_values(["absolute_change", "node"], ascending=[False, True]).reset_index(
        drop=True
    )
    return pd.DataFrame(global_rows), nodes


def spectral_embedding(graph: nx.Graph, dimensions: int = 8, seed: int = 42) -> pd.DataFrame:
    """Learn compact node representations from normalized graph structure."""

    undirected = graph.to_undirected()
    nodes = sorted(undirected.nodes(), key=str)
    n = len(nodes)
    if n < 2:
        return pd.DataFrame({"node": nodes})
    adjacency = nx.to_numpy_array(undirected, nodelist=nodes, weight="weight", dtype=float)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    n_components = max(1, min(dimensions, n - 1))
    vectors = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(normalized)
    result = pd.DataFrame(vectors, columns=[f"embedding_{i + 1}" for i in range(n_components)])
    result.insert(0, "node", nodes)
    return result


def nearest_nodes(embedding: pd.DataFrame, query_node: str, top_n: int = 5) -> pd.DataFrame:
    """Return nodes with embeddings most similar to a query node."""

    if query_node not in set(embedding["node"]):
        raise KeyError(f"Unknown node: {query_node}")
    feature_columns = [column for column in embedding.columns if column.startswith("embedding_")]
    matrix = embedding[feature_columns].to_numpy()
    similarities = cosine_similarity(matrix)
    query_index = embedding.index[embedding["node"] == query_node][0]
    result = pd.DataFrame({"node": embedding["node"], "similarity": similarities[query_index]})
    return result[result["node"] != query_node].nlargest(top_n, "similarity").reset_index(drop=True)


def top_link_predictions(graph: nx.Graph, top_n: int = 25) -> pd.DataFrame:
    """Rank absent edges with an interpretable ensemble of local predictors."""

    undirected = graph.to_undirected()
    candidates = list(nx.non_edges(undirected))
    if not candidates:
        return pd.DataFrame(
            columns=["source", "target", "score", "common_neighbors", "jaccard", "adamic_adar"]
        )
    jaccard = {(u, v): score for u, v, score in nx.jaccard_coefficient(undirected, candidates)}
    adamic = {(u, v): score for u, v, score in nx.adamic_adar_index(undirected, candidates)}
    records = []
    for u, v in candidates:
        common = len(list(nx.common_neighbors(undirected, u, v)))
        jac = jaccard.get((u, v), jaccard.get((v, u), 0.0))
        aa = adamic.get((u, v), adamic.get((v, u), 0.0))
        score = common + jac + math.log1p(aa)
        records.append(
            {
                "source": u,
                "target": v,
                "score": score,
                "common_neighbors": common,
                "jaccard": jac,
                "adamic_adar": aa,
            }
        )
    return pd.DataFrame(records).nlargest(top_n, "score").reset_index(drop=True)
