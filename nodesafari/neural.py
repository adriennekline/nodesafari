"""Dependency-light graph neural networks implemented with NumPy.

The models in this module use end-to-end graph convolutions while keeping
NodeSafari's CPU-only installation compact. They are intended for small and
medium exploratory networks, not large-scale accelerator training.
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .ml import MLInputError, _node_feature_table, _validate_labels


def _normalized_adjacency(graph: nx.Graph, nodes: list[object]) -> np.ndarray:
    """Return a symmetric, self-looped GCN propagation matrix."""

    adjacency = nx.to_numpy_array(
        graph.to_undirected(), nodelist=nodes, weight=None, dtype=float
    )
    adjacency = (adjacency != 0).astype(float) + np.eye(len(nodes))
    degree = adjacency.sum(axis=1)
    inverse = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree),
        where=degree > 0,
    )
    return inverse[:, None] * adjacency * inverse[None, :]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -40, 40))
    return exponential / exponential.sum(axis=1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def _initialize_adam(parameters: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {
        "m": {name: np.zeros_like(value) for name, value in parameters.items()},
        "v": {name: np.zeros_like(value) for name, value in parameters.items()},
    }


def _adam_step(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    state: dict[str, dict[str, np.ndarray]],
    step: int,
    learning_rate: float,
) -> None:
    beta_1, beta_2 = 0.9, 0.999
    for name in parameters:
        state["m"][name] = beta_1 * state["m"][name] + (1 - beta_1) * gradients[name]
        state["v"][name] = beta_2 * state["v"][name] + (1 - beta_2) * gradients[name] ** 2
        corrected_m = state["m"][name] / (1 - beta_1**step)
        corrected_v = state["v"][name] / (1 - beta_2**step)
        parameters[name] -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)


def _gcn_node_forward(
    propagation: np.ndarray,
    features: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    propagated_features = propagation @ features
    hidden_pre = propagated_features @ parameters["w1"] + parameters["b1"]
    hidden = np.maximum(hidden_pre, 0)
    propagated_hidden = propagation @ hidden
    logits = propagated_hidden @ parameters["w2"] + parameters["b2"]
    return logits, (propagated_features, hidden_pre, propagated_hidden)


def _train_node_gcn(
    propagation: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    *,
    class_count: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    parameters = {
        "w1": rng.normal(0, np.sqrt(2 / features.shape[1]), (features.shape[1], hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "w2": rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, class_count)),
        "b2": np.zeros(class_count),
    }
    state = _initialize_adam(parameters)
    counts = np.bincount(targets[train_indices], minlength=class_count)
    class_weights = len(train_indices) / (class_count * np.maximum(counts, 1))
    losses = []
    regularization = 1e-4

    for epoch in range(1, epochs + 1):
        logits, cache = _gcn_node_forward(propagation, features, parameters)
        probabilities = _softmax(logits)
        sample_weights = class_weights[targets[train_indices]]
        normalization = sample_weights.sum()
        chosen = np.clip(probabilities[train_indices, targets[train_indices]], 1e-9, 1)
        loss = float(-(sample_weights * np.log(chosen)).sum() / normalization)
        loss += regularization * float(
            np.square(parameters["w1"]).sum() + np.square(parameters["w2"]).sum()
        )
        losses.append(loss)

        derivative = np.zeros_like(probabilities)
        derivative[train_indices] = probabilities[train_indices]
        derivative[train_indices, targets[train_indices]] -= 1
        derivative[train_indices] *= sample_weights[:, None] / normalization

        propagated_features, hidden_pre, propagated_hidden = cache
        gradients = {
            "w2": propagated_hidden.T @ derivative + 2 * regularization * parameters["w2"],
            "b2": derivative.sum(axis=0),
        }
        hidden_derivative = propagation.T @ derivative @ parameters["w2"].T
        hidden_pre_derivative = hidden_derivative * (hidden_pre > 0)
        gradients["w1"] = (
            propagated_features.T @ hidden_pre_derivative
            + 2 * regularization * parameters["w1"]
        )
        gradients["b1"] = hidden_pre_derivative.sum(axis=0)
        _adam_step(parameters, gradients, state, epoch, learning_rate)
    return parameters, losses


def neural_node_prediction(
    graph: nx.Graph,
    labels: pd.DataFrame,
    *,
    hidden_dim: int = 16,
    epochs: int = 250,
    learning_rate: float = 0.02,
    seed: int = 42,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    """Cross-validate a two-layer GCN for transductive node classification."""

    if not {"node", "label"}.issubset(labels.columns):
        raise MLInputError("Node labels require columns named 'node' and 'label'.")
    clean = labels.copy()
    clean["node"] = clean["node"].astype(str)
    clean = clean.dropna(subset=["node", "label"]).drop_duplicates("node", keep="last")
    clean["label"] = clean["label"].astype(str)
    nodes = sorted(graph, key=str)
    node_keys = [str(node) for node in nodes]
    if len(node_keys) != len(set(node_keys)):
        raise MLInputError("Node identifiers must remain unique when represented as text.")

    feature_table = _node_feature_table(graph, clean)
    feature_table["node"] = feature_table["node"].astype(str)
    feature_table = feature_table.drop_duplicates("node").set_index("node").reindex(node_keys)
    feature_columns = list(feature_table.select_dtypes(include="number").columns)
    raw_features = feature_table[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    node_lookup = {node: index for index, node in enumerate(node_keys)}
    labeled = clean[clean["node"].isin(node_lookup)].copy()
    if len(labeled) < 4:
        raise MLInputError("At least four labeled nodes present in the network are required.")
    splits = _validate_labels(labeled["label"])
    classes = sorted(labeled["label"].astype(str).unique())
    class_lookup = {label: index for index, label in enumerate(classes)}
    labeled_indices = np.array([node_lookup[node] for node in labeled["node"]], dtype=int)
    targets = np.full(len(nodes), -1, dtype=int)
    labeled_targets = np.array(
        [class_lookup[str(label)] for label in labeled["label"]], dtype=int
    )
    targets[labeled_indices] = labeled_targets
    propagation = _normalized_adjacency(graph, nodes)
    predicted = np.empty(len(labeled), dtype=int)
    confidence = np.empty(len(labeled), dtype=float)
    histories = []
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

    for fold, (train_positions, test_positions) in enumerate(
        cv.split(labeled_indices, labeled_targets), start=1
    ):
        scaler = StandardScaler().fit(raw_features.iloc[labeled_indices[train_positions]])
        features = scaler.transform(raw_features)
        parameters, losses = _train_node_gcn(
            propagation,
            features,
            targets,
            labeled_indices[train_positions],
            class_count=len(classes),
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed + fold,
        )
        probabilities = _softmax(_gcn_node_forward(propagation, features, parameters)[0])
        held_out = labeled_indices[test_positions]
        predicted[test_positions] = probabilities[held_out].argmax(axis=1)
        confidence[test_positions] = probabilities[held_out].max(axis=1)
        histories.append(losses)

    predicted_labels = [classes[index] for index in predicted]
    results = labeled[["node", "label"]].reset_index(drop=True)
    results["predicted_label"] = predicted_labels
    results["confidence"] = confidence
    results["correct"] = results["label"].astype(str) == results["predicted_label"]
    history = pd.DataFrame(
        {"epoch": np.arange(1, epochs + 1), "training_loss": np.mean(histories, axis=0)}
    )
    metrics: dict[str, float | int | str] = {
        "model": "two-layer GCN",
        "labeled_nodes": len(results),
        "classes": len(classes),
        "cv_folds": splits,
        "accuracy": float(accuracy_score(labeled_targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labeled_targets, predicted)),
        "macro_f1": float(f1_score(labeled_targets, predicted, average="macro")),
    }
    return metrics, results, history


def _encoder_forward(
    propagation: np.ndarray,
    features: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    propagated_features = propagation @ features
    hidden_pre = propagated_features @ parameters["w1"] + parameters["b1"]
    hidden = np.maximum(hidden_pre, 0)
    propagated_hidden = propagation @ hidden
    embedding = propagated_hidden @ parameters["w2"]
    return embedding, (propagated_features, hidden_pre, propagated_hidden)


def _pair_probabilities(embedding: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    scores = (embedding[pairs[:, 0]] * embedding[pairs[:, 1]]).sum(axis=1)
    return _sigmoid(scores)


def neural_link_prediction(
    graph: nx.Graph,
    *,
    top_n: int = 25,
    embedding_dim: int = 8,
    hidden_dim: int = 16,
    epochs: int = 300,
    learning_rate: float = 0.02,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    """Train a GCN graph autoencoder and evaluate recovery of held-out edges."""

    undirected = nx.Graph(graph.to_undirected())
    nodes = sorted(undirected, key=str)
    node_lookup = {node: index for index, node in enumerate(nodes)}
    edges = list(undirected.edges())
    nonedges = list(nx.non_edges(undirected))
    if len(edges) < 8 or len(nonedges) < 4:
        raise MLInputError("Neural link prediction requires at least 8 edges and 4 absent pairs.")
    rng = np.random.default_rng(seed)
    bridges = {frozenset(edge) for edge in nx.bridges(undirected)}
    removable = [edge for edge in edges if frozenset(edge) not in bridges] or edges
    test_count = min(max(2, round(len(edges) * test_fraction)), len(removable))
    held_out = rng.choice(len(removable), size=test_count, replace=False)
    test_positive = [removable[int(index)] for index in held_out]
    train_graph = undirected.copy()
    train_graph.remove_edges_from(test_positive)

    shuffled_nonedges = [nonedges[int(index)] for index in rng.permutation(len(nonedges))]
    test_negative = shuffled_nonedges[:test_count]
    remaining_negative = shuffled_nonedges[test_count:]
    train_positive = list(train_graph.edges())
    train_negative = remaining_negative[: min(len(train_positive), len(remaining_negative))]
    if len(train_negative) < 2:
        raise MLInputError("Not enough absent pairs remain for neural model training.")

    degree = np.array([train_graph.degree(node) for node in nodes], dtype=float)
    clustering = np.array([nx.clustering(train_graph, node) for node in nodes], dtype=float)
    features = np.column_stack(
        [np.ones(len(nodes)), degree / max(degree.max(), 1), clustering]
    )
    propagation = _normalized_adjacency(train_graph, nodes)
    train_pairs_raw = train_positive + train_negative
    train_pairs = np.array(
        [[node_lookup[source], node_lookup[target]] for source, target in train_pairs_raw],
        dtype=int,
    )
    train_targets = np.array([1] * len(train_positive) + [0] * len(train_negative))

    parameters = {
        "w1": rng.normal(0, np.sqrt(2 / features.shape[1]), (features.shape[1], hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "w2": rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, embedding_dim)),
    }
    state = _initialize_adam(parameters)
    history_rows = []
    regularization = 1e-4
    for epoch in range(1, epochs + 1):
        embedding, cache = _encoder_forward(propagation, features, parameters)
        probabilities = _pair_probabilities(embedding, train_pairs)
        clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
        loss = float(
            -np.mean(train_targets * np.log(clipped) + (1 - train_targets) * np.log(1 - clipped))
        )
        loss += regularization * float(
            np.square(parameters["w1"]).sum() + np.square(parameters["w2"]).sum()
        )
        history_rows.append({"epoch": epoch, "training_loss": loss})

        score_derivative = (probabilities - train_targets) / len(train_targets)
        embedding_derivative = np.zeros_like(embedding)
        for (source, target), derivative in zip(train_pairs, score_derivative):
            embedding_derivative[source] += derivative * embedding[target]
            embedding_derivative[target] += derivative * embedding[source]
        propagated_features, hidden_pre, propagated_hidden = cache
        gradients = {
            "w2": propagated_hidden.T @ embedding_derivative
            + 2 * regularization * parameters["w2"]
        }
        hidden_derivative = propagation.T @ embedding_derivative @ parameters["w2"].T
        hidden_pre_derivative = hidden_derivative * (hidden_pre > 0)
        gradients["w1"] = (
            propagated_features.T @ hidden_pre_derivative
            + 2 * regularization * parameters["w1"]
        )
        gradients["b1"] = hidden_pre_derivative.sum(axis=0)
        _adam_step(parameters, gradients, state, epoch, learning_rate)

    embedding = _encoder_forward(propagation, features, parameters)[0]
    test_pairs_raw = test_positive + test_negative
    test_pairs = np.array(
        [[node_lookup[source], node_lookup[target]] for source, target in test_pairs_raw],
        dtype=int,
    )
    test_targets = np.array([1] * len(test_positive) + [0] * len(test_negative))
    test_probabilities = _pair_probabilities(embedding, test_pairs)
    candidate_pairs = np.array(
        [[node_lookup[source], node_lookup[target]] for source, target in nonedges], dtype=int
    )
    candidate_probabilities = _pair_probabilities(embedding, candidate_pairs)
    predictions = pd.DataFrame(nonedges, columns=["source", "target"])
    predictions["probability"] = candidate_probabilities
    predictions = predictions.sort_values("probability", ascending=False).head(top_n).reset_index(drop=True)
    metrics: dict[str, float | int | str] = {
        "model": "GCN graph autoencoder",
        "training_edges": len(train_positive),
        "held_out_edges": len(test_positive),
        "roc_auc": float(roc_auc_score(test_targets, test_probabilities)),
        "average_precision": float(average_precision_score(test_targets, test_probabilities)),
    }
    return metrics, predictions, pd.DataFrame(history_rows)


def _graph_tensor(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    nodes = sorted(graph, key=str)
    undirected = graph.to_undirected()
    degree = np.array([undirected.degree(node) for node in nodes], dtype=float)
    clustering = np.array([nx.clustering(undirected, node) for node in nodes], dtype=float)
    features = np.column_stack(
        [
            np.ones(len(nodes)),
            degree / max(degree.max(), 1),
            clustering,
            np.full(len(nodes), nx.density(undirected)),
        ]
    )
    return _normalized_adjacency(undirected, nodes), features


def _graph_gcn_forward(
    propagation: np.ndarray,
    features: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    propagated_features = propagation @ features
    hidden_1_pre = propagated_features @ parameters["w1"] + parameters["b1"]
    hidden_1 = np.maximum(hidden_1_pre, 0)
    propagated_hidden = propagation @ hidden_1
    hidden_2_pre = propagated_hidden @ parameters["w2"] + parameters["b2"]
    hidden_2 = np.maximum(hidden_2_pre, 0)
    pooled = hidden_2.mean(axis=0)
    logits = pooled @ parameters["w3"] + parameters["b3"]
    return logits, (
        propagated_features,
        hidden_1_pre,
        propagated_hidden,
        hidden_2_pre,
        hidden_2,
        pooled,
    )


def _train_graph_gcn(
    tensors: list[tuple[np.ndarray, np.ndarray]],
    targets: np.ndarray,
    train_indices: np.ndarray,
    *,
    class_count: int,
    hidden_dim: int,
    embedding_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    input_dim = tensors[0][1].shape[1]
    parameters = {
        "w1": rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "w2": rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, embedding_dim)),
        "b2": np.zeros(embedding_dim),
        "w3": rng.normal(0, np.sqrt(2 / embedding_dim), (embedding_dim, class_count)),
        "b3": np.zeros(class_count),
    }
    state = _initialize_adam(parameters)
    counts = np.bincount(targets[train_indices], minlength=class_count)
    class_weights = len(train_indices) / (class_count * np.maximum(counts, 1))
    losses = []
    regularization = 1e-4

    for epoch in range(1, epochs + 1):
        gradients = {name: np.zeros_like(value) for name, value in parameters.items()}
        loss = 0.0
        normalization = float(class_weights[targets[train_indices]].sum())
        for index in train_indices:
            propagation, features = tensors[index]
            logits, cache = _graph_gcn_forward(propagation, features, parameters)
            probabilities = _softmax(logits[None, :])[0]
            weight = class_weights[targets[index]] / normalization
            loss -= weight * np.log(np.clip(probabilities[targets[index]], 1e-9, 1))
            derivative = probabilities.copy()
            derivative[targets[index]] -= 1
            derivative *= weight

            (
                propagated_features,
                hidden_1_pre,
                propagated_hidden,
                hidden_2_pre,
                hidden_2,
                pooled,
            ) = cache
            gradients["w3"] += np.outer(pooled, derivative)
            gradients["b3"] += derivative
            pooled_derivative = derivative @ parameters["w3"].T
            hidden_2_derivative = np.broadcast_to(
                pooled_derivative / len(hidden_2), hidden_2.shape
            ).copy()
            hidden_2_pre_derivative = hidden_2_derivative * (hidden_2_pre > 0)
            gradients["w2"] += propagated_hidden.T @ hidden_2_pre_derivative
            gradients["b2"] += hidden_2_pre_derivative.sum(axis=0)
            hidden_1_derivative = (
                propagation.T @ hidden_2_pre_derivative @ parameters["w2"].T
            )
            hidden_1_pre_derivative = hidden_1_derivative * (hidden_1_pre > 0)
            gradients["w1"] += propagated_features.T @ hidden_1_pre_derivative
            gradients["b1"] += hidden_1_pre_derivative.sum(axis=0)

        for name in ("w1", "w2", "w3"):
            loss += regularization * float(np.square(parameters[name]).sum())
            gradients[name] += 2 * regularization * parameters[name]
        losses.append(float(loss))
        _adam_step(parameters, gradients, state, epoch, learning_rate)
    return parameters, losses


def neural_graph_classification(
    graphs: Mapping[str, nx.Graph],
    labels: pd.DataFrame,
    *,
    hidden_dim: int = 16,
    embedding_dim: int = 8,
    epochs: int = 250,
    learning_rate: float = 0.02,
    seed: int = 42,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    """Cross-validate a pooled two-layer GCN for whole-graph classification."""

    if not {"graph", "label"}.issubset(labels.columns):
        raise MLInputError("Graph labels require columns named 'graph' and 'label'.")
    clean = labels[["graph", "label"]].dropna().drop_duplicates("graph", keep="last")
    clean["label"] = clean["label"].astype(str)
    names = [name for name in clean["graph"] if name in graphs]
    clean = clean.set_index("graph").loc[names].reset_index()
    if len(clean) < 4:
        raise MLInputError("At least four labeled graphs are required.")
    splits = _validate_labels(clean["label"])
    classes = sorted(clean["label"].astype(str).unique())
    class_lookup = {label: index for index, label in enumerate(classes)}
    targets = np.array([class_lookup[str(label)] for label in clean["label"]], dtype=int)
    tensors = [_graph_tensor(graphs[name]) for name in clean["graph"]]
    predicted = np.empty(len(clean), dtype=int)
    confidence = np.empty(len(clean), dtype=float)
    histories = []
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

    for fold, (train_indices, test_indices) in enumerate(cv.split(np.arange(len(clean)), targets), start=1):
        parameters, losses = _train_graph_gcn(
            tensors,
            targets,
            train_indices,
            class_count=len(classes),
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed + fold,
        )
        for index in test_indices:
            probabilities = _softmax(
                _graph_gcn_forward(*tensors[index], parameters)[0][None, :]
            )[0]
            predicted[index] = probabilities.argmax()
            confidence[index] = probabilities.max()
        histories.append(losses)

    results = clean.copy()
    results["predicted_label"] = [classes[index] for index in predicted]
    results["confidence"] = confidence
    results["correct"] = results["label"].astype(str) == results["predicted_label"]
    history = pd.DataFrame(
        {"epoch": np.arange(1, epochs + 1), "training_loss": np.mean(histories, axis=0)}
    )
    metrics: dict[str, float | int | str] = {
        "model": "pooled two-layer GCN",
        "graphs": len(results),
        "classes": len(classes),
        "cv_folds": splits,
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, average="macro")),
    }
    return metrics, results, history


__all__ = [
    "neural_graph_classification",
    "neural_link_prediction",
    "neural_node_prediction",
]
