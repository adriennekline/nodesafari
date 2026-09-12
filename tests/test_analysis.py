import networkx as nx
import pandas as pd

from nodesafari.analysis import (
    compare_networks,
    network_summary,
    node_metrics,
    perturbation_screen,
    rich_club_curve,
    spectral_embedding,
    top_link_predictions,
)
from nodesafari.io import GraphInputError, graph_from_edgelist


def sample_graph():
    return nx.Graph([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])


def test_graph_loader_aggregates_duplicate_edges_and_drops_loops():
    frame = pd.DataFrame(
        {"source": ["A", "A", "A"], "target": ["B", "B", "A"], "weight": [1, 2, 99]}
    )
    graph = graph_from_edgelist(frame)
    assert graph.number_of_edges() == 1
    assert graph["A"]["B"]["weight"] == 3


def test_graph_loader_requires_source_and_target():
    try:
        graph_from_edgelist(pd.DataFrame({"from": ["A"], "to": ["B"]}))
    except GraphInputError as error:
        assert "source" in str(error)
    else:
        raise AssertionError("Expected GraphInputError")


def test_summary_and_node_metrics_are_complete():
    graph = sample_graph()
    assert network_summary(graph)["nodes"] == 4
    metrics = node_metrics(graph)
    assert set(metrics["node"]) == set(graph)
    assert metrics.iloc[0]["node"] == "C"


def test_perturbation_identifies_bridge_endpoint_as_high_impact():
    result = perturbation_screen(sample_graph())
    assert result.iloc[0]["node"] == "C"
    assert result.iloc[0]["impact_score"] > 0


def test_comparison_captures_added_degree():
    graph_a = nx.Graph([("A", "B")])
    graph_b = nx.Graph([("A", "B"), ("B", "C")])
    _, nodes = compare_networks(graph_a, graph_b)
    row = nodes[nodes["node"] == "C"].iloc[0]
    assert row["status"] == "gained"
    assert row["degree_change"] == 1


def test_ml_outputs_have_expected_shape_and_no_existing_links():
    graph = sample_graph()
    embedding = spectral_embedding(graph, dimensions=2)
    assert embedding.shape == (4, 3)
    predictions = top_link_predictions(graph)
    for row in predictions.itertuples():
        assert not graph.has_edge(row.source, row.target)


def test_rich_club_curve_has_normalized_columns():
    graph = nx.barabasi_albert_graph(20, 2, seed=1)
    curve = rich_club_curve(graph, randomizations=3, seed=1)
    assert {"degree_threshold", "observed_phi", "null_phi", "normalized_phi"} <= set(curve.columns)
