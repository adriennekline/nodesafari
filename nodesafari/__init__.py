"""NodeSafari: interpretable network discovery for biological data."""

from .analysis import (
    community_table,
    compare_networks,
    network_summary,
    node_metrics,
    perturbation_screen,
    rich_club_curve,
    spectral_embedding,
    top_link_predictions,
)
from .comparison import differential_community_analysis, differential_rich_club
from .io import GraphInputError, graph_from_edgelist, network_qc
from .ml import (
    MLInputError,
    demo_graph_dataset,
    graph_classification,
    learned_link_prediction,
    node_prediction,
)
from .neural import (
    neural_graph_classification,
    neural_link_prediction,
    neural_node_prediction,
)
from .perturbation import edge_perturbation_screen, robustness_curve
from .structure import bridge_analysis, core_periphery_table, network_statistics_table

__all__ = [
    "GraphInputError",
    "MLInputError",
    "bridge_analysis",
    "community_table",
    "compare_networks",
    "core_periphery_table",
    "demo_graph_dataset",
    "differential_community_analysis",
    "differential_rich_club",
    "edge_perturbation_screen",
    "graph_classification",
    "graph_from_edgelist",
    "learned_link_prediction",
    "network_qc",
    "network_statistics_table",
    "network_summary",
    "neural_graph_classification",
    "neural_link_prediction",
    "neural_node_prediction",
    "node_metrics",
    "node_prediction",
    "perturbation_screen",
    "rich_club_curve",
    "robustness_curve",
    "spectral_embedding",
    "top_link_predictions",
]

__version__ = "1.1.0"
