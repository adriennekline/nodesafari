"""Input validation and graph construction."""

from __future__ import annotations

import networkx as nx
import pandas as pd


class GraphInputError(ValueError):
    """Raised when an uploaded edge list cannot define a valid graph."""


def network_qc(frame: pd.DataFrame, *, directed: bool = False) -> pd.DataFrame:
    """Return a human-readable quality-control report for an edge list."""

    rows: list[dict[str, object]] = []

    def add(check: str, status: str, value: object, message: str) -> None:
        rows.append({"check": check, "status": status, "value": value, "interpretation": message})

    missing_columns = sorted({"source", "target"} - set(frame.columns))
    add(
        "Required columns",
        "error" if missing_columns else "pass",
        ", ".join(missing_columns) if missing_columns else "source, target",
        "Add the missing columns before analysis." if missing_columns else "Required columns are present.",
    )
    add("Input rows", "info", len(frame), "Number of rows in the uploaded edge list.")
    if missing_columns:
        return pd.DataFrame(rows)

    source = frame["source"]
    target = frame["target"]
    source_text = source.fillna("").astype(str).str.strip()
    target_text = target.fillna("").astype(str).str.strip()
    invalid_endpoints = int(source_text.eq("").sum() + target_text.eq("").sum())
    self_loops = int((source_text == target_text).sum() - (source_text.eq("") & target_text.eq("")).sum())

    pairs = pd.DataFrame({"source": source_text, "target": target_text})
    pairs = pairs[(pairs["source"] != "") & (pairs["target"] != "")]
    if not directed:
        normalized = pairs.apply(lambda row: tuple(sorted((row["source"], row["target"]))), axis=1)
    else:
        normalized = pairs.apply(lambda row: (row["source"], row["target"]), axis=1)
    duplicate_edges = int(normalized.duplicated().sum())

    add(
        "Missing endpoints",
        "warning" if invalid_endpoints else "pass",
        invalid_endpoints,
        "Rows with missing or blank endpoints are removed." if invalid_endpoints else "No missing endpoints detected.",
    )
    add(
        "Self-loops",
        "warning" if self_loops else "pass",
        self_loops,
        "Self-loops are removed before analysis." if self_loops else "No self-loops detected.",
    )
    add(
        "Duplicate edges",
        "warning" if duplicate_edges else "pass",
        duplicate_edges,
        "Duplicate weights are summed." if duplicate_edges else "No duplicate edges detected.",
    )

    if "weight" in frame.columns:
        numeric = pd.to_numeric(frame["weight"], errors="coerce")
        invalid = int(numeric.isna().sum() - frame["weight"].isna().sum())
        missing_weights = int(frame["weight"].isna().sum())
        nonpositive = int((numeric <= 0).sum())
        add(
            "Numeric weights",
            "error" if invalid or missing_weights else "pass",
            invalid + missing_weights,
            "Every weight must be numeric and non-missing." if invalid or missing_weights else "All weights are numeric.",
        )
        add(
            "Nonpositive weights",
            "warning" if nonpositive else "pass",
            nonpositive,
            "Some algorithms assume positive interaction strength." if nonpositive else "All weights are positive.",
        )
    else:
        add("Edge weights", "info", "unweighted", "All interactions receive weight 1.")

    try:
        graph = graph_from_edgelist(frame, directed=directed)
    except GraphInputError:
        return pd.DataFrame(rows)
    undirected = graph.to_undirected()
    components = nx.number_connected_components(undirected)
    add(
        "Connected components",
        "warning" if components > 1 else "pass",
        components,
        "Disconnected components can affect path-based metrics." if components > 1 else "The network is connected.",
    )
    add("Usable network", "pass", f"{graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges", "Network construction succeeded.")
    return pd.DataFrame(rows)


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
