"""Input validation and graph construction."""

from __future__ import annotations

import networkx as nx
import pandas as pd


class GraphInputError(ValueError):
    """Raised when an uploaded edge list cannot define a valid graph."""


def graph_from_edgelist(
    frame: pd.DataFrame,
    *,
    directed: bool = False,
    source: str = "source",
    target: str = "target",
    weight: str = "weight",
) -> nx.Graph:
    """Create a graph from a tidy edge-list DataFrame.

    Required columns are ``source`` and ``target``. ``weight`` is optional and
    defaults to 1. Duplicate edges are aggregated by summing their weights.
    Self-loops are discarded because several structural metrics used by the
    application assume a simple graph.
    """

    missing = {source, target} - set(frame.columns)
    if missing:
        raise GraphInputError(
            "Missing required column(s): " + ", ".join(sorted(missing))
        )

    clean = frame.copy()
    clean = clean.dropna(subset=[source, target])
    clean[source] = clean[source].astype(str).str.strip()
    clean[target] = clean[target].astype(str).str.strip()
    clean = clean[(clean[source] != "") & (clean[target] != "")]
    clean = clean[clean[source] != clean[target]]

    if weight not in clean.columns:
        clean[weight] = 1.0
    else:
        clean[weight] = pd.to_numeric(clean[weight], errors="coerce")
        if clean[weight].isna().any():
            raise GraphInputError("The weight column contains non-numeric values.")

    if clean.empty:
        raise GraphInputError("No valid edges remain after input validation.")

    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    for row in clean[[source, target, weight]].itertuples(index=False, name=None):
        u, v, edge_weight = row
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += float(edge_weight)
        else:
            graph.add_edge(u, v, weight=float(edge_weight))

    return graph
