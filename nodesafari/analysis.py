"""Interpretable network-science and graph-ML analyses."""

from __future__ import annotations

import math
from collections.abc import Iterable

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
    return pd.DataFrame(rows).sort_values(
        ["degree_centrality", "betweenness"], ascending=False
    ).reset_index(drop=True)


def _community_partition(graph: nx.Graph) -> list[set[str]]:
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [{str(node)} for node in graph]
    communities = nx.community.greedy_modularity_communities(
        graph.to_undirected(), weight="weight"
    )
    return [set(group) for group in communities]


def community_table(graph: nx.Graph) -> pd.DataFrame:
    """Assign nodes to deterministic greedy-modularity communities."""

    records: list[dict[str, object]] = []
    for community_id, members in enumerate(_community_partition(graph), start=1):
        for node in sorted(members, key=str):
            records.append({"node": node, "community": community_id})
    return pd.DataFrame(records)


def _rich_club_phi(graph: nx.Graph, thresholds: Iterable[int]) -> dict[int, float]:
    degrees = dict(graph.degree())
    values: dict[int, float] = {}
    for threshold in thresholds:
        members = [node for node, degree in degrees.items() if degree > threshold]
        count = len(members)
        if count < 2:
            values[threshold] = float("nan")
            continue
        edge_count = graph.subgraph(members).number_of_edges()
        values[threshold] = 2 * edge_count / (count * (count - 1))
    return values


def rich_club_curve(
    graph: nx.Graph, *, randomizations: int = 20, seed: int = 42
) -> pd.DataFrame:
    """Estimate observed and degree-preserving normalized rich-club curves.

    Normalized values above one indicate denser-than-null connectivity among
    nodes whose degree exceeds the corresponding threshold. The null ensemble
    uses degree-preserving double-edge swaps.
    """

    undirected = nx.Graph(graph.to_undirected())
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    max_degree = max((degree for _, degree in undirected.degree()), default=0)
    thresholds = list(range(max_degree))
    observed = _rich_club_phi(undirected, thresholds)
    null_values: dict[int, list[float]] = {k: [] for k in thresholds}

    rng = np.random.default_rng(seed)
    for _ in range(max(randomizations, 0)):
        null_graph = undirected.copy()
        edges = null_graph.number_of_edges()
        if edges >= 2:
            try:
                nx.double_edge_swap(
                    null_graph,
                    nswap=max(edges * 3, 1),
                    max_tries=max(edges * 100, 100),
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
            except nx.NetworkXAlgorithmError:
                pass
        phi = _rich_club_phi(null_graph, thresholds)
        for threshold, value in phi.items():
            if np.isfinite(value):
                null_values[threshold].append(value)

    rows = []
    for threshold in thresholds:
        samples = null_values[threshold]
        null_mean = float(np.mean(samples)) if samples else float("nan")
        obs = observed[threshold]
        normalized = obs / null_mean if null_mean > 0 and np.isfinite(obs) else float("nan")
        rows.append(
            {
                "degree_threshold": threshold,
                "observed_phi": obs,
                "null_phi": null_mean,
                "normalized_phi": normalized,
            }
        )
    return pd.DataFrame(rows)


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
    return pd.DataFrame(records).sort_values(
        ["impact_score", "node"], ascending=[False, True]
    ).reset_index(drop=True)


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
    centrality_a = nx.degree_centrality(graph_a.to_undirected()) if graph_a.number_of_nodes() > 1 else {}
    centrality_b = nx.degree_centrality(graph_b.to_undirected()) if graph_b.number_of_nodes() > 1 else {}
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
                "status": "gained" if node not in graph_a else "lost" if node not in graph_b else "shared",
            }
        )
    nodes = pd.DataFrame(node_rows)
    nodes["absolute_change"] = nodes["degree_change"].abs()
    nodes = nodes.sort_values(["absolute_change", "node"], ascending=[False, True]).reset_index(drop=True)
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
        return pd.DataFrame(columns=["source", "target", "score", "common_neighbors", "jaccard", "adamic_adar"])
    jaccard = {(u, v): score for u, v, score in nx.jaccard_coefficient(undirected, candidates)}
    adamic = {(u, v): score for u, v, score in nx.adamic_adar_index(undirected, candidates)}
    records = []
    for u, v in candidates:
        common = len(list(nx.common_neighbors(undirected, u, v)))
        jac = jaccard.get((u, v), jaccard.get((v, u), 0.0))
        aa = adamic.get((u, v), adamic.get((v, u), 0.0))
        score = common + jac + math.log1p(aa)
        records.append(
            {"source": u, "target": v, "score": score, "common_neighbors": common, "jaccard": jac, "adamic_adar": aa}
        )
    return pd.DataFrame(records).nlargest(top_n, "score").reset_index(drop=True)
