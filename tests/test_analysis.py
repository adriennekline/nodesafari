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
from nodesafari.comparison import differential_community_analysis, differential_rich_club
from nodesafari.io import GraphInputError, graph_from_edgelist, network_qc
from nodesafari.ml import (
    demo_graph_dataset,
    graph_classification,
    learned_link_prediction,
    node_prediction,
)
from nodesafari.neural import (
    neural_graph_classification,
    neural_link_prediction,
    neural_node_prediction,
)
from nodesafari.perturbation import edge_perturbation_screen, robustness_curve
from nodesafari.structure import bridge_analysis, core_periphery_table, network_statistics_table


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


def test_network_qc_reports_cleaning_and_connectivity():
    frame = pd.DataFrame(
        {
            "source": ["A", "A", "A", None, "C"],
            "target": ["B", "B", "A", "C", "D"],
            "weight": [1, 2, 3, 4, 5],
        }
    )
    report = network_qc(frame)
    values = report.set_index("check")["value"]
    assert values["Missing endpoints"] == 1
    assert values["Self-loops"] == 1
    assert values["Duplicate edges"] == 1
    assert values["Connected components"] == 2


def test_core_bridge_and_statistics_outputs():
    graph = sample_graph()
    cores = core_periphery_table(graph)
    bridge_nodes, bridge_edges = bridge_analysis(graph)
    statistics = network_statistics_table(graph)
    assert set(cores["node"]) == set(graph)
    assert set(bridge_nodes["node"]) == {"C"}
    assert {frozenset((row.source, row.target)) for row in bridge_edges.itertuples()} == {
        frozenset(("C", "D"))
    }
    assert "global_efficiency" in set(statistics["metric"])


def test_differential_community_and_rich_club_outputs():
    graph_a = nx.barabasi_albert_graph(20, 2, seed=1)
    graph_b = graph_a.copy()
    graph_b.remove_edge(*next(iter(graph_b.edges())))
    graph_b.add_edge(*next(nx.non_edges(graph_b)))
    summary, assignments, flows = differential_community_analysis(graph_a, graph_b)
    rich = differential_rich_club(graph_a, graph_b, randomizations=2)
    assert summary["common_nodes"] == 20
    assert len(assignments) == 20
    assert flows["nodes"].sum() == 20
    assert {"normalized_phi_a", "normalized_phi_b", "normalized_change"} <= set(rich.columns)


def test_edge_screen_and_robustness_outputs():
    graph = sample_graph()
    edges = edge_perturbation_screen(graph)
    bridge = edges[edges["is_bridge"]].iloc[0]
    assert frozenset((bridge.source, bridge.target)) == frozenset(("C", "D"))
    robustness = robustness_curve(graph, steps=3, random_repeats=3, seed=1)
    assert set(robustness["strategy"]) == {"targeted hubs", "random failure"}
    assert robustness["largest_component_mean"].between(0, 1).all()


def test_supervised_ml_workflows_return_cross_validated_outputs():
    graph = nx.barabasi_albert_graph(24, 2, seed=3)
    labels = pd.DataFrame(
        {
            "node": [str(node) for node in graph],
            "label": ["high" if node < 12 else "low" for node in graph],
        }
    )
    graph = nx.relabel_nodes(graph, str)
    node_scores, node_results, node_importance = node_prediction(graph, labels)
    link_scores, link_results, link_importance = learned_link_prediction(graph, seed=3)
    graphs, graph_labels = demo_graph_dataset(seed=3)
    graph_scores, graph_results, graph_importance = graph_classification(graphs, graph_labels)

    assert node_scores["cv_folds"] >= 2
    assert len(node_results) == 24
    assert not node_importance.empty
    assert 0 <= link_scores["roc_auc"] <= 1
    assert not link_results.empty and not link_importance.empty
    assert graph_scores["cv_folds"] >= 2
    assert len(graph_results) == 24
    assert not graph_importance.empty


def test_graph_neural_network_workflows_return_validated_outputs():
    graph = nx.barabasi_albert_graph(24, 2, seed=4)
    labels = pd.DataFrame(
        {
            "node": [str(node) for node in graph],
            "label": ["early" if node < 12 else "late" for node in graph],
        }
    )
    graph = nx.relabel_nodes(graph, str)
    node_scores, node_results, node_history = neural_node_prediction(
        graph, labels, epochs=25
    )
    link_scores, link_results, link_history = neural_link_prediction(
        graph, epochs=25, seed=4
    )
    graphs, graph_labels = demo_graph_dataset(seed=4)
    graph_scores, graph_results, graph_history = neural_graph_classification(
        graphs, graph_labels, epochs=25
    )

    assert node_scores["model"] == "two-layer GCN"
    assert len(node_results) == 24 and node_history["training_loss"].notna().all()
    assert 0 <= link_scores["average_precision"] <= 1
    assert not link_results.empty and link_history["training_loss"].notna().all()
    assert graph_scores["model"] == "pooled two-layer GCN"
    assert len(graph_results) == 24 and graph_history["training_loss"].notna().all()
    for row in link_results.itertuples():
        assert not graph.has_edge(row.source, row.target)
