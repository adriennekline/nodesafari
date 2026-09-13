"""Interpretable supervised graph-learning workflows."""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .analysis import node_metrics, spectral_embedding


class MLInputError(ValueError):
    """Raised when labels or sample counts cannot support supervised learning."""


def _validate_labels(labels: pd.Series) -> int:
    counts = labels.value_counts()
    if len(counts) < 2:
        raise MLInputError("At least two outcome classes are required.")
    if counts.min() < 2:
        raise MLInputError("Every class needs at least two labeled examples.")
    return int(min(5, counts.min()))


def _node_feature_table(graph: nx.Graph, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    metrics = node_metrics(graph).copy()
    undirected = graph.to_undirected()
    metrics["clustering"] = metrics["node"].map(nx.clustering(undirected, weight="weight"))
    embeddings = spectral_embedding(graph, dimensions=min(8, max(len(graph) - 1, 1)))
    features = metrics.merge(embeddings, on="node", how="left")
    if labels is not None:
        numeric = labels.drop(columns=[column for column in ["node", "label"] if column in labels], errors="ignore").select_dtypes(include="number")
        if not numeric.empty:
            extras = pd.concat([labels[["node"]].reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)
            features = features.merge(extras, on="node", how="left")
    return features


def node_prediction(
    graph: nx.Graph, labels: pd.DataFrame, *, seed: int = 42
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Cross-validate node-label prediction from structural and numeric features."""

    if not {"node", "label"}.issubset(labels.columns):
        raise MLInputError("Node labels require columns named 'node' and 'label'.")
    clean = labels[[column for column in labels.columns if column == "node" or column == "label" or pd.api.types.is_numeric_dtype(labels[column])]].copy()
    clean["node"] = clean["node"].astype(str)
    clean = clean.dropna(subset=["node", "label"]).drop_duplicates("node", keep="last")
    features = _node_feature_table(graph, clean).merge(clean[["node", "label"]], on="node", how="inner")
    if len(features) < 4:
        raise MLInputError("At least four labeled nodes present in the network are required.")
    splits = _validate_labels(features["label"])
    feature_columns = [
        column
        for column in features.select_dtypes(include="number").columns
        if column not in {"label"}
    ]
    x = features[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = features["label"].astype(str)
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, min_samples_leaf=1
    )
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    predicted = cross_val_predict(model, x, y, cv=cv, method="predict")
    probabilities = cross_val_predict(model, x, y, cv=cv, method="predict_proba")
    classes = sorted(y.unique())
    result = features[["node", "label"]].copy()
    result["predicted_label"] = predicted
    result["confidence"] = probabilities.max(axis=1)
    result["correct"] = result["label"].astype(str) == result["predicted_label"].astype(str)
    metrics = {
        "labeled_nodes": len(result),
        "classes": len(classes),
        "cv_folds": splits,
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
    }
    model.fit(x, y)
    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return metrics, result, importance


def _pair_features(graph: nx.Graph, pairs: list[tuple[object, object]]) -> pd.DataFrame:
    degrees = dict(graph.degree())
    rows = []
    for source, target in pairs:
        common = list(nx.common_neighbors(graph, source, target))
        union = set(graph.neighbors(source)) | set(graph.neighbors(target))
        adamic = sum(1 / np.log(max(degrees[node], 2)) for node in common)
        rows.append(
            {
                "source": source,
                "target": target,
                "common_neighbors": len(common),
                "jaccard": len(common) / len(union) if union else 0.0,
                "adamic_adar": adamic,
                "preferential_attachment": degrees[source] * degrees[target],
                "degree_sum": degrees[source] + degrees[target],
                "degree_difference": abs(degrees[source] - degrees[target]),
            }
        )
    return pd.DataFrame(rows)


def learned_link_prediction(
    graph: nx.Graph, *, top_n: int = 25, test_fraction: float = 0.2, seed: int = 42
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Train and evaluate an edge classifier, then rank absent interactions."""

    undirected = nx.Graph(graph.to_undirected())
    edges = list(undirected.edges())
    nonedges = list(nx.non_edges(undirected))
    if len(edges) < 8 or len(nonedges) < 4:
        raise MLInputError("Learned link prediction requires at least 8 edges and 4 absent node pairs.")
    rng = np.random.default_rng(seed)
    bridges = {frozenset(item) for item in nx.bridges(undirected)}
    removable = [edge for edge in edges if frozenset(edge) not in bridges]
    if not removable:
        removable = edges
    test_count = min(max(2, round(len(edges) * test_fraction)), len(removable))
    selected = rng.choice(len(removable), size=test_count, replace=False)
    test_positive = [removable[int(index)] for index in selected]
    train_graph = undirected.copy()
    train_graph.remove_edges_from(test_positive)

    nonedge_order = rng.permutation(len(nonedges))
    test_negative = [nonedges[int(index)] for index in nonedge_order[:test_count]]
    remaining_negative = [nonedges[int(index)] for index in nonedge_order[test_count:]]
    train_positive = list(train_graph.edges())
    train_negative = remaining_negative[: min(len(train_positive), len(remaining_negative))]
    if len(train_negative) < 2:
        raise MLInputError("Not enough absent pairs remain for model training.")

    train_pairs = train_positive + train_negative
    train_x = _pair_features(train_graph, train_pairs)
    train_y = np.array([1] * len(train_positive) + [0] * len(train_negative))
    feature_columns = [column for column in train_x.columns if column not in {"source", "target"}]
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, min_samples_leaf=1
    )
    model.fit(train_x[feature_columns], train_y)

    test_pairs = test_positive + test_negative
    test_x = _pair_features(train_graph, test_pairs)
    test_y = np.array([1] * len(test_positive) + [0] * len(test_negative))
    test_probability = model.predict_proba(test_x[feature_columns])[:, 1]
    metrics = {
        "training_edges": len(train_positive),
        "held_out_edges": len(test_positive),
        "roc_auc": float(roc_auc_score(test_y, test_probability)),
        "average_precision": float(average_precision_score(test_y, test_probability)),
    }

    candidates = _pair_features(train_graph, nonedges)
    candidates["probability"] = model.predict_proba(candidates[feature_columns])[:, 1]
    predictions = candidates.sort_values("probability", ascending=False).head(top_n).reset_index(drop=True)
    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return metrics, predictions, importance


def graph_feature_table(graphs: Mapping[str, nx.Graph]) -> pd.DataFrame:
    """Create fixed-length structural representations of multiple graphs."""

    rows = []
    for name, graph in graphs.items():
        undirected = graph.to_undirected()
        degrees = np.array([degree for _, degree in undirected.degree()], dtype=float)
        components = list(nx.connected_components(undirected)) if undirected else []
        largest = max((len(component) for component in components), default=0)
        assortativity = nx.degree_assortativity_coefficient(undirected) if undirected.number_of_edges() > 1 else 0.0
        rows.append(
            {
                "graph": name,
                "nodes": len(undirected),
                "edges": undirected.number_of_edges(),
                "density": nx.density(undirected),
                "average_degree": float(degrees.mean()) if len(degrees) else 0.0,
                "degree_std": float(degrees.std()) if len(degrees) else 0.0,
                "maximum_degree": float(degrees.max()) if len(degrees) else 0.0,
                "average_clustering": nx.average_clustering(undirected),
                "transitivity": nx.transitivity(undirected),
                "assortativity": float(assortativity) if np.isfinite(assortativity) else 0.0,
                "components": len(components),
                "largest_component_fraction": largest / max(len(undirected), 1),
                "global_efficiency": nx.global_efficiency(undirected) if len(undirected) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def graph_classification(
    graphs: Mapping[str, nx.Graph], labels: pd.DataFrame, *, seed: int = 42
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Cross-validate graph-level classification from structural features."""

    if not {"graph", "label"}.issubset(labels.columns):
        raise MLInputError("Graph labels require columns named 'graph' and 'label'.")
    features = graph_feature_table(graphs).merge(labels[["graph", "label"]], on="graph", how="inner")
    if len(features) < 4:
        raise MLInputError("At least four labeled graphs are required.")
    splits = _validate_labels(features["label"])
    feature_columns = [column for column in features.select_dtypes(include="number").columns]
    x = features[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = features["label"].astype(str)
    model = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=seed, min_samples_leaf=1
    )
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    predicted = cross_val_predict(model, x, y, cv=cv, method="predict")
    probabilities = cross_val_predict(model, x, y, cv=cv, method="predict_proba")
    results = features[["graph", "label"]].copy()
    results["predicted_label"] = predicted
    results["confidence"] = probabilities.max(axis=1)
    results["correct"] = results["label"].astype(str) == results["predicted_label"].astype(str)
    metrics = {
        "graphs": len(results),
        "classes": y.nunique(),
        "cv_folds": splits,
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
    }
    model.fit(x, y)
    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return metrics, results, importance


def demo_graph_dataset(seed: int = 42) -> tuple[dict[str, nx.Graph], pd.DataFrame]:
    """Generate a reproducible two-class topology dataset for the UI demonstration."""

    rng = np.random.default_rng(seed)
    graphs: dict[str, nx.Graph] = {}
    rows = []
    for index in range(12):
        n = int(rng.integers(26, 39))
        name = f"modular_{index + 1:02d}"
        graphs[name] = nx.connected_watts_strogatz_graph(n, k=4, p=float(rng.uniform(0.05, 0.20)), seed=seed + index)
        rows.append({"graph": name, "label": "modular"})
    for index in range(12):
        n = int(rng.integers(26, 39))
        name = f"hub_dominated_{index + 1:02d}"
        graphs[name] = nx.barabasi_albert_graph(n, m=2, seed=seed + 100 + index)
        rows.append({"graph": name, "label": "hub-dominated"})
    return graphs, pd.DataFrame(rows)
