"""Streamlit application for NodeSafari."""

from __future__ import annotations

import json
from base64 import b64encode
from html import escape
from importlib.metadata import PackageNotFoundError, distribution
from io import BytesIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from nodesafari.analysis import (
    community_table,
    nearest_nodes,
    network_summary,
    node_metrics,
    perturbation_screen,
    rich_club_curve,
    rich_club_edge_roles,
    rich_club_members,
    spectral_embedding,
    top_link_predictions,
)
from nodesafari.comparison import (
    compare_networks,
    differential_community_analysis,
    differential_rich_club,
)
from nodesafari.io import GraphInputError, graph_from_edgelist, network_qc
from nodesafari.ml import (
    MLInputError,
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
from nodesafari.visualization import (
    network_figure,
    rich_club_curve_figure,
    rich_club_network_figure,
)

ROOT = Path(__file__).parent
EXAMPLES = ROOT / "examples"

FA_GLYPHS = {
    "brain": "&#xf5dc;",
    "chart-line": "&#xf201;",
    "compass": "&#xf14e;",
    "database": "&#xf1c0;",
    "download": "&#xf019;",
    "exchange": "&#xf362;",
    "file-csv": "&#xf6dd;",
    "flask": "&#xf0c3;",
    "info": "&#xf05a;",
    "network": "&#xf542;",
    "shield": "&#xf3ed;",
    "sliders": "&#xf1de;",
    "upload": "&#xf093;",
}

METRIC_HELP = {
    "Nodes": "Unique entities in the active reference network after input cleaning.",
    "Interactions": "Unique edges in the active reference network after input cleaning.",
    "Modules": "Communities detected by greedy modularity optimization.",
    "Density": "Observed edges divided by the number of possible edges.",
    "Largest component": "Share of nodes contained in the largest connected component.",
    "Common nodes": "Nodes present in both the reference and comparison networks.",
    "Adjusted Rand": "Agreement between community partitions, adjusted for chance; 1 is identical.",
    "Normalized mutual info": "Shared information between community partitions; 1 is identical.",
    "Reassigned": "Share of common nodes assigned to a different aligned community.",
    "Impact": "Combined structural disruption caused by removing the selected node.",
    "Efficiency Δ": "Change in global efficiency after the selected node is removed.",
    "Components": "Connected components remaining after the selected node is removed.",
    "Labeled nodes": "Nodes with labels available for supervised evaluation.",
    "CV folds": "Number of cross-validation folds used for out-of-fold evaluation.",
    "Balanced accuracy": "Average recall across classes; 1 is best.",
    "Macro F1": "Equal-weighted F1 score across classes; 1 is best.",
    "Training edges": "Observed edges retained for model fitting.",
    "Held-out edges": "Observed edges hidden from training and used only for evaluation.",
    "ROC AUC": "Ability to rank held-out edges above absent pairs; 1 is best and 0.5 is chance.",
    "Average precision": "Precision-recall summary for held-out link recovery; 1 is best.",
    "Graphs": "Independent labeled networks included in graph-level evaluation.",
    "Rich nodes": "Nodes whose selected richness is strictly above the inspected threshold.",
    "Rich edges": "Edges connecting two rich-club members at the inspected threshold.",
    "Empirical p": "One-sided plus-one empirical p-value from the null-network ensemble.",
    "BH q-value": "Benjamini-Hochberg adjusted value reported descriptively across nested thresholds.",
}

METRIC_ICONS = {
    "Nodes": "network",
    "Interactions": "exchange",
    "Modules": "database",
    "Density": "chart-line",
    "Largest component": "shield",
    "Common nodes": "network",
    "Adjusted Rand": "exchange",
    "Normalized mutual info": "chart-line",
    "Reassigned": "exchange",
    "Impact": "flask",
    "Efficiency Δ": "chart-line",
    "Components": "network",
    "Labeled nodes": "database",
    "CV folds": "exchange",
    "Balanced accuracy": "chart-line",
    "Macro F1": "chart-line",
    "Training edges": "network",
    "Held-out edges": "shield",
    "ROC AUC": "chart-line",
    "Average precision": "chart-line",
    "Graphs": "database",
    "Rich nodes": "network",
    "Rich edges": "exchange",
    "Empirical p": "chart-line",
    "BH q-value": "shield",
}

DOWNLOAD_HELP = {
    "network_a_qc.csv": "Download the complete quality-control report for reference network A.",
    "network_b_qc.csv": "Download the complete quality-control report for comparison network B.",
    "nodesafari_analysis_manifest.csv": "Download the active input provenance and analysis settings for reproducibility.",
    "node_metrics.csv": "Download centrality and structural metrics for every node.",
    "rich_club_curve.csv": "Download observed, null, normalized, and threshold-level rich-club evidence for the selected richness measure.",
    "rich_club_membership.csv": "Download every node's richness and membership status at the selected threshold.",
    "rich_club_edge_roles.csv": "Download rich-club, feeder, and local edge classifications at the selected threshold.",
    "communities.csv": "Download the detected community assignment for every node.",
    "core_periphery.csv": "Download k-core numbers and core-periphery roles for every node.",
    "articulation_nodes.csv": "Download nodes whose removal increases network fragmentation.",
    "bridge_edges.csv": "Download edges whose removal increases network fragmentation.",
    "network_statistics.csv": "Download the whole-network statistics and their interpretations.",
    "differential_nodes.csv": "Download node-level changes between reference A and comparison B.",
    "differential_communities.csv": "Download aligned community assignments and reassignment status.",
    "differential_rich_club.csv": "Download both normalized rich-club curves and their differences.",
    "node_perturbation_screen.csv": "Download the ranked results of every single-node deletion.",
    "edge_perturbation_screen.csv": "Download the ranked results of every single-edge deletion.",
    "robustness_curve.csv": "Download targeted and random removal robustness trajectories.",
    "node_embeddings.csv": "Download the learned embedding coordinates for every node.",
    "node_predictions.csv": "Download out-of-fold node predictions and probabilities.",
    "learned_link_predictions.csv": "Download ranked candidate links and predicted probabilities.",
    "graph_predictions.csv": "Download out-of-fold graph predictions and probabilities.",
}

DATA_PROFILES = {
    "one_network": {
        "label": "One network",
        "capabilities": {"network"},
    },
    "two_networks": {
        "label": "Two networks or conditions",
        "capabilities": {"network", "comparison"},
    },
    "node_labels": {
        "label": "One network with node labels",
        "capabilities": {"network", "node_labels"},
    },
    "graph_labels": {
        "label": "Multiple networks with graph labels",
        "capabilities": {"network", "comparison", "graph_labels"},
    },
    "demo": {
        "label": "I want to learn with the demo data",
        "capabilities": {"network", "comparison", "node_labels", "graph_labels"},
    },
}

NAVIGATOR_GOALS = {
    "organize": {
        "question": "What organizes this network?",
        "title": "Map network structure",
        "primary": "Community detection",
        "supporting": "Hubs and centrality, k-core structure, bridges, and global statistics",
        "why": "Use complementary structural views to find modules, central nodes, and connectors without assuming labels.",
        "requires": {"network"},
        "needs": "One edge-list network",
        "workspace": "Explore",
        "path": "Explore → Overview & hubs, Communities, Core & bridges",
        "icon": "compass",
        "caveat": "Detected structure is descriptive and should be interpreted with domain knowledge.",
    },
    "rich_club": {
        "question": "Do highly connected nodes form an unusually dense core?",
        "title": "Test rich-club organization",
        "primary": "Normalized rich-club analysis",
        "supporting": "Degree centrality and k-core structure",
        "why": "Compare the observed connectivity among high-degree nodes with degree-preserving null networks.",
        "requires": {"network"},
        "needs": "One sufficiently connected edge-list network",
        "workspace": "Explore",
        "path": "Explore → Rich club",
        "icon": "network",
        "caveat": "Small or sparse networks can produce unstable normalized coefficients.",
    },
    "compare": {
        "question": "What changed between two networks or conditions?",
        "title": "Run differential network analysis",
        "primary": "Differential hubs and whole-network statistics",
        "supporting": "Differential communities and differential rich-club curves",
        "why": "Separate global reorganization from node-level changes and shifts in modular structure.",
        "requires": {"network", "comparison"},
        "needs": "Two edge-list networks with comparable node identities",
        "workspace": "Compare",
        "path": "Compare → Network & hubs, Communities, Rich club",
        "icon": "exchange",
        "caveat": "A two-network contrast is descriptive; replicated observations are needed for population-level inference.",
    },
    "critical": {
        "question": "Which nodes or connections are structurally critical?",
        "title": "Simulate network disruption",
        "primary": "Node and edge deletion screens",
        "supporting": "Bridge analysis and targeted-versus-random robustness",
        "why": "Rank elements by the structural damage caused by removing them and examine network resilience.",
        "requires": {"network"},
        "needs": "One edge-list network",
        "workspace": "Perturb",
        "path": "Perturb → Remove nodes, Remove edges, Robustness",
        "icon": "flask",
        "caveat": "Structural impact does not establish causal or experimental importance.",
    },
    "similar": {
        "question": "Which nodes have similar structural roles?",
        "title": "Compare node representations",
        "primary": "Spectral node embeddings",
        "supporting": "Nearest-node search and community assignments",
        "why": "Represent each node by its network context and rank nodes occupying similar structural positions.",
        "requires": {"network"},
        "needs": "One edge-list network",
        "workspace": "Predict",
        "path": "Predict → Node embeddings",
        "icon": "chart-line",
        "caveat": "Embedding similarity suggests a shared network role, not necessarily shared function.",
    },
    "node_prediction": {
        "question": "Can network information predict labels for nodes?",
        "title": "Evaluate node classification",
        "primary": "Class-balanced random forest baseline",
        "supporting": "Two-layer graph convolutional network",
        "why": "Establish an interpretable baseline, then test whether a GCN adds useful signal from topology.",
        "requires": {"network", "node_labels"},
        "needs": "One network plus a node-and-label CSV with at least two classes",
        "workspace": "Predict",
        "path": "Predict → Node prediction",
        "icon": "brain",
        "caveat": "Use out-of-fold scores for model comparison and reserve external data for final validation.",
    },
    "link_prediction": {
        "question": "Which connections may be missing?",
        "title": "Rank candidate links",
        "primary": "Random forest link-prediction baseline",
        "supporting": "GCN graph autoencoder",
        "why": "Evaluate recovery of hidden observed edges before ranking currently absent connections.",
        "requires": {"network"},
        "needs": "One network with enough observed and absent node pairs",
        "workspace": "Predict",
        "path": "Predict → Link prediction",
        "icon": "network",
        "caveat": "A high score prioritizes validation; it is not evidence that a connection exists.",
    },
    "graph_prediction": {
        "question": "Can I classify entire networks?",
        "title": "Evaluate graph classification",
        "primary": "Random forest graph-level baseline",
        "supporting": "Pooled two-layer graph convolutional network",
        "why": "Test whether whole-network structure distinguishes graph-level classes across independent samples.",
        "requires": {"network", "graph_labels"},
        "needs": "Multiple independent edge lists plus graph-level labels with at least two samples per class",
        "workspace": "Predict",
        "path": "Predict → Graph classification",
        "icon": "brain",
        "caveat": "Independent networks—not nodes from one network—must form the evaluation samples.",
    },
}


@st.cache_resource
def fontawesome_font_css() -> str:
    """Embed Font Awesome Free so icons also work in offline Docker sessions."""

    try:
        package = distribution("fontawesome-free")
        font_path = Path(
            package.locate_file(
                "fontawesome-free/static/fontawesome_free/webfonts/fa-solid-900.woff2"
            )
        )
        encoded = b64encode(font_path.read_bytes()).decode("ascii")
    except (PackageNotFoundError, OSError):
        return ""
    return (
        "@font-face{font-family:'Font Awesome 5 Free';font-style:normal;font-weight:900;"
        "font-display:block;src:url(data:font/woff2;base64,"
        f"{encoded}) format('woff2');}}"
    )


def fa_icon(name: str, tooltip: str, class_name: str = "") -> str:
    """Return an accessible, hover-explained Font Awesome icon."""

    glyph = FA_GLYPHS[name]
    return (
        f'<span class="fa-icon {escape(class_name)}" role="img" tabindex="0" '
        f'aria-label="{escape(tooltip)}" data-tooltip="{escape(tooltip)}">{glyph}</span>'
    )


st.set_page_config(page_title="NodeSafari · Network Discovery", page_icon="🧭", layout="wide")
st.markdown(
    "<style>"
    + fontawesome_font_css()
    + """
    :root {
      --ink:#201a3b; --panel:#ffffff; --purple:#7c3aed; --purple-dark:#5b21b6;
      --teal:#0f9f91; --teal-dark:#0f766e; --muted:#68637d; --line:#e7e2f1;
      --lavender:#f7f5fc; --mint:#f0fdfa;
    }
    .stApp {
      background:
        radial-gradient(circle at 88% 0%,rgba(45,212,191,.07),transparent 26%),
        radial-gradient(circle at 22% 0%,rgba(139,92,246,.06),transparent 28%),
        #ffffff;
      color:#201a3b;
    }
    [data-testid="stHeader"] { background:rgba(255,255,255,.82); backdrop-filter:blur(10px); }
    [data-testid="stSidebar"] { background:#faf9ff; border-right:1px solid #e5def5; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height:1.45; }
    .block-container { max-width:1480px; padding-top:1.35rem; padding-bottom:3rem; }
    h1,h2,h3 { letter-spacing:-0.025em; }
    h1 { font-size:2.35rem !important; margin:.15rem 0 .35rem !important; }
    h2 { margin-top:.25rem !important; }
    .brand-lockup { display:flex; align-items:center; gap:.7rem; margin:.1rem 0 1.3rem; }
    .brand-mark {
      width:2.35rem; height:2.35rem; border-radius:12px; display:grid; place-items:center;
      background:linear-gradient(135deg,#7c3aed,#0f9f91); color:white; font-size:1.25rem;
      box-shadow:0 8px 18px rgba(91,33,182,.18);
    }
    .fa-icon {
      position:relative; display:inline-flex; align-items:center; justify-content:center;
      font-family:"Font Awesome 5 Free"; font-style:normal; font-weight:900;
      speak:none;
    }
    .fa-icon[data-tooltip]::after {
      content:attr(data-tooltip); position:absolute; z-index:9999; left:50%; bottom:calc(100% + .55rem);
      transform:translateX(-50%) translateY(3px); width:max-content; max-width:260px;
      padding:.45rem .6rem; border-radius:8px; background:#241d40; color:#fff;
      font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      font-size:.72rem; font-weight:520; line-height:1.35; letter-spacing:0;
      box-shadow:0 8px 22px rgba(36,29,64,.2); opacity:0; visibility:hidden;
      pointer-events:none; transition:opacity .14s ease,transform .14s ease;
    }
    .fa-icon[data-tooltip]:hover::after,.fa-icon[data-tooltip]:focus::after {
      opacity:1; visibility:visible; transform:translateX(-50%) translateY(0);
    }
    .brand-name { color:#201a3b; font-size:1.08rem; font-weight:760; letter-spacing:-.02em; }
    .brand-version { color:#807a91; font-size:.72rem; font-weight:650; letter-spacing:.08em; text-transform:uppercase; }
    .hero {
      padding:1rem 1.25rem .95rem; margin-bottom:.75rem; border:1px solid var(--line);
      border-radius:18px; background:rgba(255,255,255,.9);
      box-shadow:0 16px 42px rgba(54,35,93,.07);
    }
    .eyebrow { color:#0f9f91; font-size:.74rem; font-weight:750; letter-spacing:.16em; text-transform:uppercase; }
    .hero-title { color:#201a3b; font-size:2rem; font-weight:770; letter-spacing:-.045em; line-height:1.08; margin:.22rem 0 .32rem; }
    .lede { color:#625d76; font-size:.94rem; max-width:900px; margin:0; line-height:1.48; }
    .workspace-kicker { color:#827b93; font-size:.68rem; font-weight:760; letter-spacing:.14em; text-transform:uppercase; margin:.9rem 0 .3rem; }
    .context-strip { display:flex; flex-wrap:wrap; align-items:center; gap:.5rem; margin:.15rem 0 1.15rem; }
    .context-label { color:#827b93; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-right:.1rem; }
    .context-chip { color:#4f4762; background:white; border:1px solid #ded8ea; border-radius:999px; padding:.26rem .62rem; font-size:.76rem; }
    .context-chip strong { color:#30294e; }
    .context-chip.demo { background:#f5f3ff; border-color:#ddd6fe; color:#6d28d9; }
    .context-chip.upload { background:#ecfdf9; border-color:#99f6e4; color:#0f766e; }
    .navigator-card {
      border:1px solid #ddd6ee; border-radius:15px; background:linear-gradient(135deg,#ffffff,#fbfaff);
      padding:1rem 1.05rem; box-shadow:0 8px 22px rgba(55,35,100,.05); margin:.2rem 0 .75rem;
    }
    .navigator-heading { display:flex; gap:.65rem; align-items:flex-start; }
    .navigator-heading .fa-icon { flex:0 0 2rem; width:2rem; height:2rem; border-radius:9px; color:#fff; background:linear-gradient(135deg,#7c3aed,#0f9f91); }
    .navigator-title { color:#261f42; font-size:1.02rem; font-weight:740; line-height:1.25; }
    .navigator-primary { color:#0f766e; font-size:.78rem; font-weight:700; margin-top:.2rem; }
    .navigator-grid { display:grid; grid-template-columns:1fr 1fr; gap:.65rem 1.2rem; margin-top:.9rem; }
    .navigator-item { color:#625d76; font-size:.79rem; line-height:1.45; }
    .navigator-item b { display:block; color:#827b93; font-size:.66rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.12rem; }
    .navigator-route { color:#5b21b6; background:#f2efff; border:1px solid #ddd6fe; border-radius:9px; padding:.55rem .7rem; margin-top:.8rem; font-size:.79rem; font-weight:680; }
    .navigator-caveat { color:#716b84; font-size:.75rem; line-height:1.42; margin-top:.65rem; }
    .navigator-placeholder { color:#716b84; font-size:.84rem; padding:.75rem .1rem .2rem; }
    .start-steps { display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; margin:.2rem 0 1.1rem; }
    .start-step { border:1px solid #e4deef; border-radius:12px; background:#fff; padding:.8rem .9rem; color:#6b647d; font-size:.78rem; line-height:1.42; }
    .start-step b { display:block; color:#30294e; font-size:.84rem; margin-bottom:.2rem; }
    .step-number { display:inline-grid; place-items:center; width:1.4rem; height:1.4rem; border-radius:50%; background:#ede9fe; color:#6d28d9; font-weight:760; margin-right:.35rem; }
    @media (max-width:800px) { .navigator-grid,.start-steps { grid-template-columns:1fr; } }
    [data-testid="stMetric"] {
      background:linear-gradient(145deg,#ffffff,#fbfaff);
      border:1px solid #e2dcf2;
      box-shadow:0 8px 24px rgba(55,35,100,.055),inset 0 3px 0 rgba(45,212,191,.65);
      padding:.88rem 1rem;
      border-radius:14px; min-height:92px;
    }
    .metric-fa { position:relative; z-index:2; height:0; top:.78rem; margin-left:calc(100% - 2.15rem); color:#7c3aed; }
    .metric-fa .fa-icon { width:1.35rem; height:1.35rem; border-radius:7px; background:#f2efff; font-size:.7rem; }
    [data-testid="stDownloadButton"] button::before {
      content:"\\f019"; font-family:"Font Awesome 5 Free"; font-weight:900;
      color:#0f9f91; margin-right:.42rem;
    }
    .stTabs [data-baseweb="tab"]::before {
      content:"\\f14e"; font-family:"Font Awesome 5 Free"; font-weight:900;
      color:#0f9f91; margin-right:.38rem; font-size:.78rem;
    }
    [data-testid="stMetricLabel"] { color:#68637d; font-weight:600; }
    [data-testid="stMetricValue"] { color:#241d40; letter-spacing:-.035em; }
    .section-kicker { color:#0f9f91; font-size:.7rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.2rem; }
    .section-title { color:#261f42; font-size:1.32rem; font-weight:740; letter-spacing:-.025em; margin:0 0 .22rem; }
    .section-description { color:#716b84; font-size:.9rem; line-height:1.5; margin:0 0 1rem; max-width:900px; }
    .insight { border-left:3px solid #14b8a6; background:linear-gradient(90deg,#ecfdf9,#f7f5ff); padding:.85rem 1rem; border-radius:0 12px 12px 0; color:#30294e; line-height:1.5; }
    .empty-state { border:1px dashed #cfc5e8; background:#fbfaff; padding:1.35rem; border-radius:14px; color:#625d76; }
    .empty-state strong { color:#30294e; display:block; margin-bottom:.25rem; }
    .source-card { border:1px solid #e4deef; background:#ffffff; border-radius:12px; padding:.72rem .8rem; margin:.45rem 0 .9rem; }
    .source-card .label { color:#847d95; font-size:.68rem; font-weight:750; letter-spacing:.1em; text-transform:uppercase; }
    .source-row { display:grid; grid-template-columns:1.25rem 1fr; gap:.35rem; align-items:start; padding-top:.52rem; }
    .source-row + .source-row { margin-top:.45rem; border-top:1px solid #eeeaf5; }
    .source-key { color:#0f9f91; font-size:.72rem; font-weight:800; padding-top:.08rem; }
    .source-card .value { color:#30294e; font-size:.8rem; font-weight:680; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .source-card .meta { color:#817a91; font-size:.69rem; margin-top:.08rem; }
    .method-note { color:#716b84; font-size:.84rem; }
    .stTabs [data-baseweb="tab-list"] { gap:.32rem; border-bottom:1px solid #e9e4f1; padding-bottom:.42rem; }
    .stTabs [data-baseweb="tab"] { background:transparent; border:1px solid transparent; border-radius:10px; padding:.52rem .78rem; color:#6b647d; font-weight:650; }
    .stTabs [aria-selected="true"] { background:linear-gradient(135deg,#ede9fe,#e6fffb); border-color:#c4b5fd; color:#5b21b6; }
    [data-testid="stDataFrame"] { border:1px solid #e7e2f1; border-radius:12px; overflow:hidden; }
    .stButton button,.stDownloadButton button { background:#ffffff; border-color:#8b5cf6; color:#6d28d9; border-radius:9px; font-weight:620; }
    .stButton button:hover,.stDownloadButton button:hover { background:#f0fdfa; border-color:#0f9f91; color:#0f766e; }
    [data-testid="stFileUploaderDropzone"] { background:#fdfcff; border-color:#cfc5e8; border-radius:12px; }
    [data-testid="stExpander"] { border-color:#e5dfef; border-radius:12px; background:#fff; }
    .sidebar-foot { color:#8a8498; font-size:.72rem; line-height:1.45; margin-top:1.2rem; }
    .app-footer { color:#8a8498; font-size:.76rem; text-align:center; padding-top:.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_csv(uploaded, fallback: Path) -> pd.DataFrame:
    return pd.read_csv(uploaded if uploaded is not None else fallback)


@st.cache_data
def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def graph_cache_signature(graph) -> tuple[object, ...]:
    """Return a stable cache key that retains node identities and edge weights."""

    nodes = tuple(sorted(repr(node) for node in graph.nodes()))
    edges = tuple(
        sorted(
            (
                repr(source),
                repr(target),
                float(data.get("weight", 1.0)),
            )
            for source, target, data in graph.edges(data=True)
        )
    )
    return (graph.is_directed(), nodes, edges)


@st.cache_data(show_spinner=False)
def cached_rich_club_curve(
    _graph,
    graph_signature: tuple[object, ...],
    randomizations: int,
    seed: int,
    swaps_per_edge: int,
    min_rich_nodes: int,
    richness: str,
    weighted: bool,
) -> pd.DataFrame:
    """Cache expensive null ensembles across display-only Streamlit reruns."""

    del graph_signature
    return rich_club_curve(
        _graph,
        randomizations=randomizations,
        seed=seed,
        swaps_per_edge=swaps_per_edge,
        min_rich_nodes=min_rich_nodes,
        richness=richness,
        weighted=weighted,
    )


@st.cache_data(show_spinner=False)
def cached_differential_rich_club(
    _graph_a,
    _graph_b,
    graph_a_signature: tuple[object, ...],
    graph_b_signature: tuple[object, ...],
    randomizations: int,
    seed: int,
    swaps_per_edge: int,
    min_rich_nodes: int,
    richness: str,
    weighted: bool,
) -> pd.DataFrame:
    """Cache paired null ensembles across display-only Streamlit reruns."""

    del graph_a_signature, graph_b_signature
    return differential_rich_club(
        _graph_a,
        _graph_b,
        randomizations=randomizations,
        seed=seed,
        swaps_per_edge=swaps_per_edge,
        min_rich_nodes=min_rich_nodes,
        richness=richness,
        weighted=weighted,
    )


def download_csv_button(label: str, frame: pd.DataFrame, file_name: str) -> None:
    """Render a consistently explained CSV export control."""

    st.download_button(
        label,
        csv_bytes(frame),
        file_name,
        "text/csv",
        help=DOWNLOAD_HELP[file_name],
        on_click="ignore",
    )


def figure_svg_bytes(figure) -> bytes:
    """Serialize a Matplotlib figure as a scalable vector graphic."""

    buffer = BytesIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    return buffer.getvalue()


def rich_club_methods_text(
    *,
    richness: str,
    weighted: bool,
    randomizations: int,
    swaps_per_edge: int,
    min_rich_nodes: int,
    seed: int,
) -> str:
    """Generate editable reporting text from the active rich-club settings."""

    coefficient = (
        "an Opsahl-style weighted rich-club coefficient"
        if weighted
        else "the binary rich-club coefficient"
    )
    weight_note = (
        " Edge weights were randomly permuted over rewired edges, preserving the global "
        "weight distribution but not node strength."
        if weighted
        else ""
    )
    return (
        f"Rich-club organization was evaluated across {richness} thresholds using "
        f"{coefficient}. The observed network was compared with {randomizations} null "
        "networks generated by degree-preserving double-edge swaps "
        f"({swaps_per_edge} attempted swaps per edge; random seed {seed}).{weight_note} "
        "Normalized coefficients were calculated as the observed coefficient divided by "
        "the mean null coefficient. Empirical one-sided p-values used a plus-one correction, "
        "and Benjamini-Hochberg q-values were reported descriptively. Thresholds retaining "
        f"fewer than {min_rich_nodes} rich nodes were flagged as unstable and excluded from "
        "exploratory-signal designation. Analyses were performed with NodeSafari v1.5.0."
    )


def styled(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(gridColor="#e8e3f2", labelColor="#625d76", titleColor="#30294e")
        .configure_legend(labelColor="#625d76", titleColor="#30294e")
        .configure_view(strokeOpacity=0)
    )


def metric_row(values: list[tuple[str, object, str | None]]) -> None:
    columns = st.columns(len(values))
    for column, (label, value, help_text) in zip(columns, values):
        explanation = help_text or METRIC_HELP.get(label, f"Summary metric: {label}.")
        icon_name = METRIC_ICONS.get(label, "chart-line")
        column.markdown(
            f'<div class="metric-fa">{fa_icon(icon_name, explanation)}</div>',
            unsafe_allow_html=True,
        )
        column.metric(label, value, help=explanation)


def section_heading(title: str, description: str, kicker: str | None = None) -> None:
    icons = {
        "Getting started": "compass",
        "Validate": "shield",
        "Explore": "compass",
        "Compare": "exchange",
        "Perturb": "flask",
        "Predict": "brain",
    }
    kicker_icon = fa_icon(icons.get(kicker or "", "info"), description, "section-fa")
    kicker_html = (
        f'<div class="section-kicker">{kicker_icon} {escape(kicker)}</div>' if kicker else ""
    )
    st.markdown(
        f'{kicker_html}<div class="section-title">{escape(title)}</div>'
        f'<div class="section-description">{escape(description)}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, description: str) -> None:
    st.markdown(
        f'<div class="empty-state"><strong>{fa_icon("info", description)} '
        f"{escape(title)}</strong>{escape(description)}</div>",
        unsafe_allow_html=True,
    )


def active_data_card(
    reference: str,
    reference_metadata: str,
    comparison: str,
    comparison_metadata: str,
) -> None:
    """Show both active network sources in one compact sidebar card."""

    st.markdown(
        f'<div class="source-card"><div class="label">{fa_icon("database", "Active network sources")} '
        "Active data</div>"
        '<div class="source-row"><div class="source-key">A</div><div>'
        f'<div class="value">{escape(reference)}</div>'
        f'<div class="meta">{escape(reference_metadata)}</div></div></div>'
        '<div class="source-row"><div class="source-key">B</div><div>'
        f'<div class="value">{escape(comparison)}</div>'
        f'<div class="meta">{escape(comparison_metadata)}</div></div></div></div>',
        unsafe_allow_html=True,
    )


def navigator_card(recommendation: dict[str, object]) -> None:
    """Render a method recommendation with inputs, rationale, route, and caveat."""

    description = str(recommendation["why"])
    st.markdown(
        '<div class="navigator-card"><div class="navigator-heading">'
        f"{fa_icon(str(recommendation['icon']), description)}<div>"
        f'<div class="navigator-title">{escape(str(recommendation["title"]))}</div>'
        f'<div class="navigator-primary">Primary method · '
        f"{escape(str(recommendation['primary']))}</div></div></div>"
        '<div class="navigator-grid">'
        f'<div class="navigator-item"><b>Why this fits</b>{escape(description)}</div>'
        f'<div class="navigator-item"><b>Supporting methods</b>'
        f"{escape(str(recommendation['supporting']))}</div>"
        f'<div class="navigator-item"><b>Required inputs</b>'
        f"{escape(str(recommendation['needs']))}</div>"
        f'<div class="navigator-item"><b>Recommended order</b>Start with the primary method, '
        "then use the supporting analyses to check whether the result is consistent.</div></div>"
        f'<div class="navigator-route">Recommended route · '
        f"{escape(str(recommendation['path']))}</div>"
        f'<div class="navigator-caveat">{fa_icon("info", str(recommendation["caveat"]))} '
        f"{escape(str(recommendation['caveat']))}</div></div>",
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        f'<div class="brand-lockup"><div class="brand-mark">'
        f"{fa_icon('compass', 'NodeSafari network discovery workspace')}</div><div>"
        '<div class="brand-name">NodeSafari</div><div class="brand-version">Research workspace · v1.5.0</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Data workspace")
    st.caption("Bring an edge list, or explore the complete workflow with synthetic examples.")
    uploaded_a = st.file_uploader(
        "Network A · reference",
        type="csv",
        help="Upload the reference edge list. Required columns: source and target; weight is optional.",
    )
    uploaded_b = st.file_uploader(
        "Network B · optional comparison",
        type="csv",
        help="Upload an optional second edge list to enable differential network analyses.",
    )
    primary_is_demo = uploaded_a is None
    comparison_is_demo = uploaded_b is None and primary_is_demo
    reference_value = "Synthetic reference example" if primary_is_demo else uploaded_a.name
    reference_metadata = "Demo data" if primary_is_demo else "Uploaded CSV"
    if uploaded_b is not None:
        comparison_value = uploaded_b.name
        comparison_metadata = "Uploaded CSV"
    elif comparison_is_demo:
        comparison_value = "Synthetic comparison example"
        comparison_metadata = "Demo data"
    else:
        comparison_value = "Not loaded"
        comparison_metadata = "Upload B to enable comparison"
    active_data_card(
        reference_value,
        reference_metadata,
        comparison_value,
        comparison_metadata,
    )

    with st.expander("Network settings", expanded=False):
        directed = st.toggle(
            "Directed network",
            value=False,
            help="Enable when edge direction matters, such as regulatory or flow networks.",
        )
    with st.expander("Rich-club settings", expanded=False):
        randomizations = st.select_slider(
            "Rich-club null networks",
            options=[10, 25, 50, 100, 250, 500, 1000],
            value=100,
            help="Use 1,000 for final inference; smaller ensembles are intended for rapid exploration.",
        )
        rich_weighted = st.toggle(
            "Weighted rich-club coefficient",
            value=False,
            help="Use edge weights in the rich-club coefficient. If the input has no weight column, all edges have unit weight.",
        )
        richness = st.selectbox(
            "Richness measure",
            ["degree", "strength"] if rich_weighted else ["degree"],
            help="Degree counts connections. Strength sums edge weights and is available only with the weighted coefficient.",
        )
        swaps_per_edge = st.slider(
            "Rich-club swaps per edge",
            1,
            25,
            10,
            help="Attempted double-edge swaps per edge in each degree-preserving null network.",
        )
        min_rich_nodes = st.slider(
            "Minimum rich nodes",
            3,
            20,
            5,
            help="Thresholds retaining fewer nodes are flagged as unstable and excluded from signal designation.",
        )
        rich_club_seed = st.number_input(
            "Rich-club random seed",
            min_value=0,
            value=42,
            step=1,
            help="Fixed seed used to reproduce the null-network ensemble.",
        )
        st.caption("Higher null counts improve stability but take longer.")
    with st.expander("Perturbation settings", expanded=False):
        robustness_repeats = st.slider(
            "Random robustness repeats",
            10,
            100,
            30,
            step=10,
            help="Number of repeated random-failure simulations used to estimate the robustness band.",
        )
    with st.expander("CSV format", expanded=False):
        st.code("source,target,weight\nTP53,MDM2,1.0", language="text")
        st.caption("The weight column is optional. Node IDs can be text or numbers.")
    st.markdown(
        '<div class="sidebar-foot">Uploads remain in the active Streamlit session. '
        "NodeSafari does not write them to a persistent application volume.<br><br>"
        '<a href="https://github.com/adriennekline/nodesafari" target="_blank">GitHub</a> · '
        '<a href="https://github.com/adriennekline/nodesafari/blob/main/docs/METHODS.md" '
        'target="_blank">Methods &amp; limitations</a></div>',
        unsafe_allow_html=True,
    )


try:
    edges_a = read_csv(uploaded_a, EXAMPLES / "control_edges.csv")
    qc_a = network_qc(edges_a, directed=directed)
    graph_a = graph_from_edgelist(edges_a, directed=directed)
    if uploaded_b is not None:
        edges_b = read_csv(uploaded_b, EXAMPLES / "disease_edges.csv")
    elif comparison_is_demo:
        edges_b = read_csv(None, EXAMPLES / "disease_edges.csv")
    else:
        edges_b = None
    qc_b = network_qc(edges_b, directed=directed) if edges_b is not None else None
    graph_b = graph_from_edgelist(edges_b, directed=directed) if edges_b is not None else None
except (GraphInputError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(str(error))
    st.stop()

if rich_weighted:
    weighted_graphs = [("reference network A", graph_a)]
    if graph_b is not None:
        weighted_graphs.append(("comparison network B", graph_b))
    invalid_weight_sources = [
        label
        for label, graph in weighted_graphs
        if any(float(data.get("weight", 1.0)) < 0 for _, _, data in graph.edges(data=True))
    ]
    if invalid_weight_sources:
        st.error(
            "Weighted rich-club analysis requires non-negative edge weights. Correct "
            + " and ".join(invalid_weight_sources)
            + " or turn off the weighted coefficient."
        )
        st.stop()

st.markdown(
    '<div class="hero"><div class="eyebrow">NodeSafari · interpretable network discovery</div>'
    '<div class="hero-title">Turn connected data into testable questions.</div>'
    '<div class="lede">Choose a workspace below to validate data, discover structure, '
    "compare conditions, test perturbations, or evaluate predictions.</div></div>",
    unsafe_allow_html=True,
)

primary_source_name = "Synthetic demo" if primary_is_demo else uploaded_a.name
comparison_source_name = (
    "Synthetic demo"
    if comparison_is_demo
    else uploaded_b.name
    if uploaded_b is not None
    else "Not loaded"
)
primary_source = escape(primary_source_name)
comparison_source = escape(comparison_source_name)
primary_class = "demo" if primary_is_demo else "upload"
comparison_class = "demo" if comparison_is_demo else "upload" if uploaded_b else ""
summary = network_summary(graph_a)
communities = community_table(graph_a)
metrics = node_metrics(graph_a)

st.markdown('<div class="workspace-kicker">Applications</div>', unsafe_allow_html=True)
getting_started_tab, qc_tab, explore_tab, compare_tab, perturb_tab, ml_tab = st.tabs(
    ["Getting Started", "Data & QC", "Explore", "Compare", "Perturb", "Predict"]
)

with getting_started_tab:
    section_heading(
        "Start with your question",
        "The Analysis Navigator translates your goal and available data into a recommended method, supporting checks, and a clear workspace route.",
        "Getting started",
    )
    st.markdown(
        '<div class="start-steps">'
        '<div class="start-step"><b><span class="step-number">1</span>Choose data</b>'
        "Use the demo or upload network A in the sidebar. Add network B only for comparison.</div>"
        '<div class="start-step"><b><span class="step-number">2</span>Choose a question</b>'
        "Describe what you want to learn; you do not need to know an algorithm name.</div>"
        '<div class="start-step"><b><span class="step-number">3</span>Open the route</b>'
        "Use the recommended application tab and supporting analyses to check the result.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    navigator_left, navigator_right = st.columns([1.25, 1], gap="large")
    with navigator_left:
        navigator_goal = st.selectbox(
            "What do you want to learn?",
            options=list(NAVIGATOR_GOALS),
            index=None,
            format_func=lambda key: NAVIGATOR_GOALS[key]["question"],
            placeholder="Choose the question closest to yours",
            help="Select the scientific question first. NodeSafari will recommend a method rather than asking you to choose an algorithm by name.",
        )
    with navigator_right:
        navigator_profiles = list(DATA_PROFILES)
        default_profile = (
            "demo" if primary_is_demo else "two_networks" if graph_b is not None else "one_network"
        )
        navigator_data = st.selectbox(
            "What data do you have?",
            options=navigator_profiles,
            index=navigator_profiles.index(default_profile),
            format_func=lambda key: DATA_PROFILES[key]["label"],
            help="Choose the description of your intended dataset, even if you have not uploaded every file yet.",
        )

    if navigator_goal is None:
        st.markdown(
            '<div class="navigator-placeholder">Choose the question closest to yours to see '
            "a recommended analysis route. Every application remains available in the tabs above.</div>",
            unsafe_allow_html=True,
        )
    else:
        recommendation = NAVIGATOR_GOALS[navigator_goal]
        navigator_card(recommendation)
        capabilities = DATA_PROFILES[navigator_data]["capabilities"]
        missing = set(recommendation["requires"]) - set(capabilities)
        if missing:
            missing_labels = {
                "network": "an edge-list network",
                "comparison": "a second comparable network",
                "node_labels": "node-level labels",
                "graph_labels": "multiple independent networks with graph-level labels",
            }
            st.warning(
                "Your selected question needs "
                + ", ".join(missing_labels[item] for item in sorted(missing))
                + ". You can still inspect the recommended workspace and use the demo while preparing those inputs."
            )
        else:
            st.success("Your stated data support the recommended analysis.")

        if navigator_goal == "compare" and graph_b is None:
            st.info("Upload comparison network B in the sidebar to activate this workspace.")
        elif navigator_goal == "node_prediction" and not primary_is_demo:
            st.info("Upload the node-label CSV inside Predict → Node prediction.")
        elif navigator_goal == "graph_prediction" and navigator_data != "demo":
            st.info(
                "Upload the independent graph edge lists and graph-label CSV inside "
                "Predict → Graph classification."
            )

        compatible = [
            str(details["question"])
            for key, details in NAVIGATOR_GOALS.items()
            if key != navigator_goal and set(details["requires"]) <= set(capabilities)
        ]
        with st.popover("Other questions supported by these data"):
            for question in compatible:
                st.markdown(f"- {question}")
    st.divider()
    section_heading(
        "Active data at a glance",
        "Confirm what is loaded before moving into an analysis application.",
        "Validate",
    )
    st.markdown(
        '<div class="context-strip"><span class="context-label">Active analysis</span>'
        f'<span class="context-chip {primary_class}"><strong>A</strong> · {primary_source}</span>'
        f'<span class="context-chip {comparison_class}"><strong>B</strong> · {comparison_source}</span>'
        f'<span class="context-chip">{"Directed" if directed else "Undirected"}</span>'
        f'<span class="context-chip">{"Weighted" if "weight" in edges_a.columns else "Unweighted"}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    metric_row(
        [
            ("Nodes", f"{summary['nodes']:.0f}", None),
            ("Interactions", f"{summary['edges']:.0f}", None),
            ("Modules", f"{communities['community'].nunique():.0f}", None),
            ("Density", f"{summary['density']:.3f}", None),
            ("Largest component", f"{summary['largest_component_fraction']:.0%}", None),
        ]
    )

with qc_tab:
    section_heading(
        "Input quality control",
        "Review cleaning decisions and connectivity before interpreting downstream results.",
        "Validate",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Reference network (A)**")
        st.dataframe(qc_a.assign(value=qc_a["value"].astype(str)), hide_index=True, width="stretch")
        download_csv_button("Download A QC report", qc_a, "network_a_qc.csv")
    with right:
        st.markdown("**Comparison network (B)**")
        if qc_b is None:
            empty_state(
                "No comparison network loaded",
                "Upload network B in the sidebar when you are ready to run differential analyses.",
            )
        else:
            st.dataframe(
                qc_b.assign(value=qc_b["value"].astype(str)), hide_index=True, width="stretch"
            )
            download_csv_button("Download B QC report", qc_b, "network_b_qc.csv")
    st.markdown(
        '<p class="method-note">Warnings identify analyzable conditions that may change '
        "interpretation. Errors must be corrected before graph construction.</p>",
        unsafe_allow_html=True,
    )
    manifest = pd.DataFrame(
        [
            {"setting": "NodeSafari version", "value": "1.5.0"},
            {"setting": "Reference source", "value": primary_source_name},
            {"setting": "Comparison source", "value": comparison_source_name},
            {"setting": "Graph type", "value": "directed" if directed else "undirected"},
            {
                "setting": "Weight mode",
                "value": "weighted" if "weight" in edges_a.columns else "unweighted",
            },
            {"setting": "Reference nodes", "value": str(int(summary["nodes"]))},
            {"setting": "Reference edges", "value": str(int(summary["edges"]))},
            {"setting": "Rich-club null networks", "value": str(randomizations)},
            {"setting": "Rich-club richness", "value": richness},
            {
                "setting": "Rich-club coefficient",
                "value": "weighted" if rich_weighted else "binary",
            },
            {"setting": "Rich-club swaps per edge", "value": str(swaps_per_edge)},
            {"setting": "Minimum rich nodes", "value": str(min_rich_nodes)},
            {"setting": "Rich-club random seed", "value": str(int(rich_club_seed))},
            {"setting": "Robustness repeats", "value": str(robustness_repeats)},
        ]
    )
    with st.expander("Analysis manifest", expanded=False):
        st.caption("Save the active inputs and settings alongside exported results.")
        st.dataframe(manifest, hide_index=True, width="stretch")
        download_csv_button(
            "Download analysis manifest", manifest, "nodesafari_analysis_manifest.csv"
        )

with explore_tab:
    overview_subtab, rich_subtab, community_subtab, core_subtab, stats_subtab = st.tabs(
        ["Overview & hubs", "Rich club", "Communities", "Core & bridges", "Statistics"]
    )

    with overview_subtab:
        section_heading(
            "Network overview",
            "See modular organization and the nodes that occupy prominent structural roles.",
            "Explore",
        )
        left, right = st.columns([1.45, 1], gap="large")
        with left:
            st.pyplot(network_figure(graph_a, communities), width="stretch")
        with right:
            st.markdown("**High-interest nodes**")
            display = metrics[["node", "degree", "betweenness", "closeness", "pagerank"]].head(12)
            st.dataframe(display, hide_index=True, width="stretch")
            top_node = str(display.iloc[0]["node"])
            st.markdown(
                f'<div class="insight"><b>{top_node}</b> is the leading degree hub. '
                "Centrality is a structural signal, not evidence of causality.</div>",
                unsafe_allow_html=True,
            )
            download_csv_button("Download node metrics", metrics, "node_metrics.csv")

    with rich_subtab:
        section_heading(
            "Rich-club organization",
            "Test enrichment against a degree-preserving null ensemble, then inspect the nodes and edge roles behind the curve.",
            "Explore",
        )
        if directed:
            st.info(
                "Rich-club analysis uses the undirected projection of this directed network. "
                "Interpret the result as connectivity, not directional flow."
            )
        curve = cached_rich_club_curve(
            graph_a,
            graph_cache_signature(graph_a),
            randomizations,
            int(rich_club_seed),
            swaps_per_edge,
            min_rich_nodes,
            richness,
            rich_weighted,
        )
        for warning in curve.attrs.get("warnings", []):
            st.warning(warning)

        evidence_tab, members_tab, export_tab = st.tabs(
            ["Evidence", "Membership & roles", "Reproducible export"]
        )
        threshold_title = f"{richness.capitalize()} threshold"
        with evidence_tab:
            chart_frame = curve.dropna(subset=["rho"])
            if chart_frame.empty:
                st.info("This graph is too small or sparse for a stable normalized curve.")
            else:
                left, right = st.columns(2, gap="large")
                with left:
                    st.markdown("**Observed coefficient and null envelope**")
                    band = (
                        alt.Chart(curve)
                        .mark_area(color="#C4B5FD", opacity=0.32)
                        .encode(
                            x=alt.X("threshold:Q", title=threshold_title),
                            y=alt.Y("phi_null_lower_95:Q", title="Rich-club coefficient φ"),
                            y2="phi_null_upper_95:Q",
                            tooltip=[
                                "threshold",
                                "n_rich_nodes",
                                "phi_observed",
                                "phi_null_mean",
                                "phi_null_lower_95",
                                "phi_null_upper_95",
                            ],
                        )
                    )
                    line_data = curve.melt(
                        id_vars=["threshold"],
                        value_vars=["phi_observed", "phi_null_mean"],
                        var_name="series",
                        value_name="coefficient",
                    )
                    lines = (
                        alt.Chart(line_data)
                        .mark_line(point=True, strokeWidth=2.5)
                        .encode(
                            x=alt.X("threshold:Q", title=threshold_title),
                            y=alt.Y("coefficient:Q", title="Rich-club coefficient φ"),
                            color=alt.Color(
                                "series:N",
                                scale=alt.Scale(
                                    domain=["phi_observed", "phi_null_mean"],
                                    range=["#0F9F91", "#7C3AED"],
                                ),
                                legend=alt.Legend(title=None),
                            ),
                            tooltip=["threshold", "series", "coefficient"],
                        )
                    )
                    st.altair_chart(styled((band + lines).properties(height=330)), width="stretch")
                with right:
                    st.markdown("**Normalized enrichment and threshold evidence**")
                    normalized = (
                        alt.Chart(chart_frame)
                        .mark_line(point=True, color="#0F9F91", strokeWidth=2.8)
                        .encode(
                            x=alt.X("threshold:Q", title=threshold_title),
                            y=alt.Y("rho:Q", title="Normalized coefficient ρ"),
                            tooltip=[
                                "threshold",
                                "n_rich_nodes",
                                "n_rich_edges",
                                "rho",
                                "p_empirical",
                                "q_bh",
                                "reliable_node_count",
                            ],
                        )
                    )
                    reference = (
                        alt.Chart(pd.DataFrame({"rho": [1.0]}))
                        .mark_rule(color="#7C3AED", strokeDash=[5, 5])
                        .encode(y="rho:Q")
                    )
                    signals = (
                        alt.Chart(chart_frame[chart_frame["exploratory_signal"]])
                        .mark_point(color="#F97316", filled=True, size=85)
                        .encode(x="threshold:Q", y="rho:Q", tooltip=["threshold", "rho"])
                    )
                    unreliable = (
                        alt.Chart(chart_frame[~chart_frame["reliable_node_count"]])
                        .mark_point(color="#9CA3AF", filled=False, size=75)
                        .encode(x="threshold:Q", y="rho:Q", tooltip=["threshold", "n_rich_nodes"])
                    )
                    st.altair_chart(
                        styled(
                            (normalized + reference + signals + unreliable).properties(height=330)
                        ),
                        width="stretch",
                    )
                reliable = chart_frame[chart_frame["reliable_node_count"]]
                signal_count = int(reliable["exploratory_signal"].sum())
                peak_source = reliable if not reliable.empty else chart_frame
                peak = peak_source.loc[peak_source["rho"].idxmax()]
                st.markdown(
                    f'<div class="insight">Peak reliable enrichment occurs above '
                    f"<b>{richness} {peak.threshold:g}</b> (ρ = <b>{peak.rho:.2f}</b>; "
                    f"{int(peak.n_rich_nodes)} nodes retained). <b>{signal_count}</b> reliable "
                    "thresholds meet the exploratory ρ &gt; 1 and empirical p &lt; 0.05 rule.</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<p class="method-note">A value above one is not sufficient by itself. '
                "Look for persistence across meaningful thresholds, the full null distribution, "
                "retained node counts, and sensitivity to network construction.</p>",
                unsafe_allow_html=True,
            )
            with st.expander("Threshold-level evidence table", expanded=False):
                st.dataframe(
                    curve[
                        [
                            "threshold",
                            "n_rich_nodes",
                            "n_rich_edges",
                            "phi_observed",
                            "phi_null_mean",
                            "phi_null_lower_95",
                            "phi_null_upper_95",
                            "rho",
                            "p_empirical",
                            "q_bh",
                            "reliable_node_count",
                            "exploratory_signal",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )
            download_csv_button("Download rich-club evidence", curve, "rich_club_curve.csv")

        with members_tab:
            eligible = curve.loc[curve["reliable_node_count"], "threshold"].tolist()
            if not eligible:
                empty_state(
                    "No reliable membership threshold",
                    "Lower the minimum-rich-node setting or use a larger, denser network.",
                )
            else:
                selected_threshold = st.selectbox(
                    "Inspect threshold",
                    eligible,
                    index=max(0, len(eligible) // 2),
                    help="Inspect the exact nodes and edge roles underlying one reliable point on the curve.",
                )
                members = rich_club_members(graph_a, selected_threshold, richness=richness)
                score_weight = "weight" if richness == "strength" else None
                membership = pd.DataFrame(
                    [
                        {
                            "node": node,
                            "richness": float(
                                graph_a.to_undirected().degree(node, weight=score_weight)
                            ),
                            "rich_club_member": node in members,
                        }
                        for node in graph_a
                    ]
                ).sort_values(["rich_club_member", "richness"], ascending=[False, False])
                edge_roles = rich_club_edge_roles(graph_a, selected_threshold, richness=richness)
                selected = curve.loc[curve["threshold"] == selected_threshold].iloc[0]
                metric_row(
                    [
                        (
                            "Rich nodes",
                            f"{int(selected.n_rich_nodes)}",
                            "Nodes with richness strictly above the selected threshold.",
                        ),
                        (
                            "Rich edges",
                            f"{int(selected.n_rich_edges)}",
                            "Edges connecting two rich-club members at the selected threshold.",
                        ),
                        (
                            "Empirical p",
                            f"{selected.p_empirical:.3f}",
                            "One-sided plus-one empirical p-value from the null ensemble.",
                        ),
                        (
                            "BH q-value",
                            f"{selected.q_bh:.3f}",
                            "Benjamini-Hochberg adjusted value, reported descriptively because thresholds are nested.",
                        ),
                    ]
                )
                visual, tables = st.columns([1.25, 1], gap="large")
                with visual:
                    role_figure = rich_club_network_figure(
                        graph_a,
                        selected_threshold,
                        richness=richness,
                        seed=int(rich_club_seed),
                    )
                    st.pyplot(role_figure, width="stretch")
                    st.download_button(
                        "Download role network SVG",
                        figure_svg_bytes(role_figure),
                        "rich_club_roles.svg",
                        "image/svg+xml",
                        help="Download a scalable vector figure of membership and edge roles at this threshold.",
                        on_click="ignore",
                    )
                with tables:
                    st.markdown("**Node membership**")
                    st.dataframe(membership, hide_index=True, width="stretch", height=220)
                    download_csv_button(
                        "Download membership", membership, "rich_club_membership.csv"
                    )
                    st.markdown("**Edge roles**")
                    st.dataframe(edge_roles, hide_index=True, width="stretch", height=220)
                    download_csv_button(
                        "Download edge roles", edge_roles, "rich_club_edge_roles.csv"
                    )

        with export_tab:
            methods_text = rich_club_methods_text(
                richness=richness,
                weighted=rich_weighted,
                randomizations=randomizations,
                swaps_per_edge=swaps_per_edge,
                min_rich_nodes=min_rich_nodes,
                seed=int(rich_club_seed),
            )
            st.text_area(
                "Generated Methods text",
                methods_text,
                height=185,
                help="Editable reporting language generated from the exact active analysis settings.",
            )
            settings = {
                "software": "NodeSafari",
                "version": "1.5.0",
                "source": primary_source_name,
                "directed_input_projected_to_undirected": directed,
                "analysis": curve.attrs.get("parameters", {}),
                "network": summary,
            }
            left, middle, right = st.columns(3)
            left.download_button(
                "Download Methods text",
                methods_text,
                "rich_club_methods.txt",
                "text/plain",
                help="Download editable Methods language for this exact run.",
                on_click="ignore",
            )
            publication_figure = rich_club_curve_figure(curve, richness=richness)
            middle.download_button(
                "Download curve SVG",
                figure_svg_bytes(publication_figure),
                "rich_club_curve.svg",
                "image/svg+xml",
                help="Download a publication-oriented scalable vector figure of both rich-club panels.",
                on_click="ignore",
            )
            right.download_button(
                "Download settings JSON",
                json.dumps(settings, indent=2, sort_keys=True),
                "rich_club_settings.json",
                "application/json",
                help="Download analysis parameters, source context, and the network summary.",
                on_click="ignore",
            )

    with community_subtab:
        section_heading(
            "Detected communities",
            "Inspect densely connected modules and export assignments for annotation workflows.",
            "Explore",
        )
        community_sizes = communities.groupby("community").size().reset_index(name="nodes")
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            community_chart = (
                alt.Chart(community_sizes)
                .mark_bar(color="#2DD4BF", cornerRadiusEnd=4)
                .encode(
                    x=alt.X("nodes:Q", title="Nodes"),
                    y=alt.Y("community:N", sort="-x", title="Community"),
                    tooltip=["community", "nodes"],
                )
                .properties(height=320)
            )
            st.altair_chart(styled(community_chart), width="stretch")
        with right:
            st.dataframe(communities, hide_index=True, width="stretch", height=320)
            download_csv_button("Download communities", communities, "communities.csv")
        st.markdown(
            '<p class="method-note">Communities use greedy modularity optimization and '
            "should be interpreted alongside relevant domain annotation.</p>",
            unsafe_allow_html=True,
        )

    with core_subtab:
        cores = core_periphery_table(graph_a)
        bridge_nodes, bridge_edges = bridge_analysis(graph_a)
        section_heading(
            "K-core and bridge structure",
            "Separate densely embedded nodes from the periphery and identify exact disconnection points.",
            "Explore",
        )
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.markdown("**Core–periphery position**")
            st.dataframe(cores, hide_index=True, width="stretch", height=380)
            download_csv_button("Download core positions", cores, "core_periphery.csv")
        with right:
            st.markdown("**Articulation nodes**")
            if bridge_nodes.empty:
                st.success("No articulation nodes were found.")
            else:
                st.dataframe(bridge_nodes, hide_index=True, width="stretch", height=170)
            st.markdown("**Bridge edges**")
            if bridge_edges.empty:
                st.success("No bridge edges were found.")
            else:
                st.dataframe(bridge_edges, hide_index=True, width="stretch", height=170)
            download_csv_button("Download bridge nodes", bridge_nodes, "articulation_nodes.csv")
            download_csv_button("Download bridge edges", bridge_edges, "bridge_edges.csv")
        st.markdown(
            '<p class="method-note">Core position uses k-core decomposition. Bridges and '
            "articulation nodes are exact disconnection points in the undirected projection.</p>",
            unsafe_allow_html=True,
        )

    with stats_subtab:
        statistics = network_statistics_table(graph_a)
        section_heading(
            "Whole-network statistics",
            "A compact profile of connectivity, distance, clustering, and degree structure.",
            "Explore",
        )
        st.dataframe(statistics, hide_index=True, width="stretch")
        download_csv_button("Download network statistics", statistics, "network_statistics.csv")

with compare_tab:
    if graph_b is None:
        section_heading(
            "Compare networks",
            "Measure structural changes between a reference network and a second condition.",
            "Compare",
        )
        empty_state(
            "A second network is required",
            "Upload comparison network B in the sidebar. NodeSafari will never mix your uploaded reference with demo comparison data.",
        )
    else:
        global_subtab, community_compare_subtab, rich_compare_subtab = st.tabs(
            ["Network & hubs", "Communities", "Rich club"]
        )

        with global_subtab:
            section_heading(
                "Reference A → comparison B",
                "Quantify whole-network changes and rank the largest node-level degree shifts.",
                "Compare",
            )
            global_comparison, node_comparison = compare_networks(graph_a, graph_b)
            first, second = st.columns([0.9, 1.35], gap="large")
            with first:
                st.markdown("**Whole-network changes**")
                st.dataframe(global_comparison, hide_index=True, width="stretch")
            with second:
                st.markdown("**Largest degree shifts**")
                shift_chart = (
                    alt.Chart(node_comparison.head(15))
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("degree_change:Q", title="Degree change"),
                        y=alt.Y("node:N", sort="-x", title=None),
                        color=alt.condition(
                            alt.datum.degree_change >= 0,
                            alt.value("#2DD4BF"),
                            alt.value("#8B5CF6"),
                        ),
                        tooltip=["node", "degree_a", "degree_b", "degree_change"],
                    )
                    .properties(height=390)
                )
                st.altair_chart(styled(shift_chart), width="stretch")
            download_csv_button(
                "Download differential nodes", node_comparison, "differential_nodes.csv"
            )

        with community_compare_subtab:
            section_heading(
                "Differential communities",
                "Track module stability, reassignment, and overlap between the two networks.",
                "Compare",
            )
            community_summary, node_assignments, community_flows = differential_community_analysis(
                graph_a, graph_b
            )
            metric_row(
                [
                    ("Common nodes", community_summary["common_nodes"], None),
                    (
                        "Adjusted Rand",
                        f"{community_summary['adjusted_rand_index']:.3f}",
                        "1 means identical partitions.",
                    ),
                    (
                        "Normalized mutual info",
                        f"{community_summary['normalized_mutual_information']:.3f}",
                        "1 means identical partitions.",
                    ),
                    (
                        "Reassigned",
                        f"{community_summary['fraction_reassigned']:.0%}",
                        "After overlap alignment.",
                    ),
                ]
            )
            left, right = st.columns([1.2, 1], gap="large")
            with left:
                st.markdown("**Node-level assignments**")
                st.dataframe(node_assignments, hide_index=True, width="stretch", height=360)
            with right:
                st.markdown("**Community flows**")
                st.dataframe(community_flows, hide_index=True, width="stretch", height=360)
            download_csv_button(
                "Download differential communities",
                node_assignments,
                "differential_communities.csv",
            )
            st.markdown(
                '<p class="method-note">B communities are aligned to A by maximal node overlap '
                "before reassignment is reported.</p>",
                unsafe_allow_html=True,
            )

        with rich_compare_subtab:
            section_heading(
                "Differential rich-club organization",
                f"Compare normalized rich-club profiles across matched {richness} thresholds.",
                "Compare",
            )
            rich_difference = cached_differential_rich_club(
                graph_a,
                graph_b,
                graph_cache_signature(graph_a),
                graph_cache_signature(graph_b),
                randomizations,
                int(rich_club_seed),
                swaps_per_edge,
                min_rich_nodes,
                richness,
                rich_weighted,
            )
            long_curve = rich_difference.melt(
                id_vars="degree_threshold",
                value_vars=["normalized_phi_a", "normalized_phi_b"],
                var_name="network",
                value_name="normalized_phi",
            ).dropna()
            long_curve["network"] = long_curve["network"].map(
                {"normalized_phi_a": "Network A", "normalized_phi_b": "Network B"}
            )
            if long_curve.empty:
                st.info("The networks are too small or sparse for comparable normalized curves.")
            else:
                rich_chart = (
                    alt.Chart(long_curve)
                    .mark_line(point=True, strokeWidth=3)
                    .encode(
                        x=alt.X("degree_threshold:Q", title=f"{richness.capitalize()} threshold"),
                        y=alt.Y("normalized_phi:Q", title="Normalized rich-club coefficient"),
                        color=alt.Color(
                            "network:N",
                            scale=alt.Scale(range=["#2DD4BF", "#8B5CF6"]),
                            title=None,
                        ),
                        tooltip=["network", "degree_threshold", "normalized_phi"],
                    )
                    .properties(height=360)
                )
                st.altair_chart(styled(rich_chart), width="stretch")
            st.dataframe(rich_difference, hide_index=True, width="stretch", height=260)
            download_csv_button(
                "Download differential rich club",
                rich_difference,
                "differential_rich_club.csv",
            )

with perturb_tab:
    node_perturb_subtab, edge_perturb_subtab, robustness_subtab = st.tabs(
        ["Remove nodes", "Remove edges", "Robustness"]
    )

    with node_perturb_subtab:
        perturbations = perturbation_screen(graph_a)
        section_heading(
            "Single-node deletion screen",
            "Rank nodes by the structural disruption caused by removing them one at a time.",
            "Perturb",
        )
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            perturb_chart = (
                alt.Chart(perturbations.head(15))
                .mark_bar(color="#8B5CF6", cornerRadiusEnd=4)
                .encode(
                    x=alt.X("impact_score:Q", title="Structural impact score"),
                    y=alt.Y("node:N", sort="-x", title=None),
                    tooltip=["node", "impact_score", "efficiency_change", "components_after"],
                )
                .properties(height=420)
            )
            st.altair_chart(styled(perturb_chart), width="stretch")
        with right:
            selected_node = st.selectbox(
                "Inspect a node deletion",
                perturbations["node"],
                help="Select a node to inspect its simulated structural impact in detail.",
            )
            selected_result = perturbations.loc[perturbations["node"] == selected_node].iloc[0]
            metric_row(
                [
                    ("Impact", f"{selected_result['impact_score']:.4f}", None),
                    ("Efficiency Δ", f"{selected_result['efficiency_change']:.4f}", None),
                    ("Components", int(selected_result["components_after"]), None),
                ]
            )
            st.dataframe(perturbations, hide_index=True, width="stretch", height=310)
        download_csv_button("Download node screen", perturbations, "node_perturbation_screen.csv")

    with edge_perturb_subtab:
        edge_perturbations = edge_perturbation_screen(graph_a)
        section_heading(
            "Single-edge deletion screen",
            "Find connections whose loss most changes global efficiency or fragments the network.",
            "Perturb",
        )
        st.dataframe(edge_perturbations, hide_index=True, width="stretch", height=430)
        download_csv_button(
            "Download edge screen", edge_perturbations, "edge_perturbation_screen.csv"
        )
        st.markdown(
            '<p class="method-note">Impact combines global-efficiency loss and loss of the '
            "largest component. Bridge status is reported separately.</p>",
            unsafe_allow_html=True,
        )

    with robustness_subtab:
        section_heading(
            "Network robustness",
            "Compare targeted hub removal with repeated random failures across increasing damage levels.",
            "Perturb",
        )
        robustness = robustness_curve(graph_a, random_repeats=robustness_repeats)
        plot = robustness.copy()
        plot["lower"] = (plot["largest_component_mean"] - plot["largest_component_std"]).clip(0)
        plot["upper"] = (plot["largest_component_mean"] + plot["largest_component_std"]).clip(
            upper=1
        )
        random_plot = plot[plot["strategy"] == "random failure"]
        band = (
            alt.Chart(random_plot)
            .mark_area(color="#2DD4BF", opacity=0.18)
            .encode(x="fraction_removed:Q", y="lower:Q", y2="upper:Q")
        )
        lines = (
            alt.Chart(plot)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X(
                    "fraction_removed:Q",
                    title="Fraction of nodes removed",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("largest_component_mean:Q", title="Largest component / original nodes"),
                color=alt.Color(
                    "strategy:N", scale=alt.Scale(range=["#2DD4BF", "#8B5CF6"]), title=None
                ),
                tooltip=["strategy", "fraction_removed", "largest_component_mean"],
            )
            .properties(height=390)
        )
        st.altair_chart(styled(band + lines), width="stretch")
        download_csv_button("Download robustness curve", robustness, "robustness_curve.csv")

with ml_tab:
    embedding_subtab, node_ml_subtab, link_ml_subtab, graph_ml_subtab = st.tabs(
        ["Node embeddings", "Node prediction", "Link prediction", "Graph classification"]
    )

    with embedding_subtab:
        embeddings = spectral_embedding(graph_a, dimensions=8)
        section_heading(
            "Node embeddings",
            "Map nodes into a compact coordinate space so similar structural roles sit near one another.",
            "Predict",
        )
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.markdown("**Spectral node map**")
            if {"embedding_1", "embedding_2"}.issubset(embeddings.columns):
                plotted = embeddings.merge(communities, on="node", how="left")
                scatter = (
                    alt.Chart(plotted)
                    .mark_circle(size=115, opacity=0.86)
                    .encode(
                        x=alt.X("embedding_1:Q", title="Embedding dimension 1"),
                        y=alt.Y("embedding_2:Q", title="Embedding dimension 2"),
                        color=alt.Color("community:N", legend=alt.Legend(title="Module")),
                        tooltip=["node", "community"],
                    )
                    .properties(height=390)
                )
                st.altair_chart(styled(scatter), width="stretch")
        with right:
            query_node = st.selectbox(
                "Find nodes with a similar network role",
                list(embeddings["node"]),
                help="Select a node to rank other nodes by cosine similarity in embedding space.",
            )
            st.dataframe(
                nearest_nodes(embeddings, query_node), hide_index=True, width="stretch", height=390
            )
        download_csv_button("Download embeddings", embeddings, "node_embeddings.csv")

    with node_ml_subtab:
        section_heading(
            "Node-label prediction",
            "Evaluate whether node features and network structure predict known labels using out-of-fold estimates.",
            "Predict",
        )
        node_model = st.radio(
            "Model",
            ["Random forest", "Graph neural network"],
            horizontal=True,
            key="node_model",
            captions=[
                "Interpretable baseline using engineered node metrics.",
                "Two-layer GCN learning from topology and available labels.",
            ],
            help="Choose the model family. Compare validated scores rather than training loss.",
        )
        node_label_upload = st.file_uploader(
            "Node labels",
            type="csv",
            help="Required: node, label. Numeric columns become features.",
            key="node_labels",
        )
        if node_label_upload is None and not primary_is_demo:
            empty_state(
                "Labels are needed for your uploaded network",
                "Upload a CSV with node and label columns. Demo labels are only used with the demo network.",
            )
        else:
            node_labels = read_csv(node_label_upload, EXAMPLES / "node_labels.csv")
            if node_label_upload is None:
                st.caption("Using synthetic demo labels · replace them with your own outcomes")
            try:
                if node_model == "Graph neural network":
                    node_scores, node_results, node_diagnostic = neural_node_prediction(
                        graph_a, node_labels
                    )
                else:
                    node_scores, node_results, node_diagnostic = node_prediction(
                        graph_a, node_labels
                    )
                st.caption(f"Model: {node_scores.get('model', 'class-balanced random forest')}")
                metric_row(
                    [
                        ("Labeled nodes", node_scores["labeled_nodes"], None),
                        ("CV folds", node_scores["cv_folds"], None),
                        (
                            "Balanced accuracy",
                            f"{node_scores['balanced_accuracy']:.3f}",
                            "Average recall across classes; 1 is best.",
                        ),
                        (
                            "Macro F1",
                            f"{node_scores['macro_f1']:.3f}",
                            "Equal-weighted F1 across classes; 1 is best.",
                        ),
                    ]
                )
                left, right = st.columns([1.25, 1], gap="large")
                with left:
                    st.dataframe(node_results, hide_index=True, width="stretch", height=350)
                with right:
                    if node_model == "Graph neural network":
                        loss_chart = (
                            alt.Chart(node_diagnostic)
                            .mark_line(color="#2DD4BF", strokeWidth=3)
                            .encode(
                                x=alt.X("epoch:Q", title="Training epoch"),
                                y=alt.Y(
                                    "training_loss:Q",
                                    title="Mean cross-validation training loss",
                                ),
                                tooltip=["epoch", "training_loss"],
                            )
                            .properties(height=350)
                        )
                        st.altair_chart(styled(loss_chart), width="stretch")
                    else:
                        importance_chart = (
                            alt.Chart(node_diagnostic.head(12))
                            .mark_bar(color="#2DD4BF", cornerRadiusEnd=4)
                            .encode(
                                x=alt.X("importance:Q", title="Random-forest importance"),
                                y=alt.Y("feature:N", sort="-x", title=None),
                                tooltip=["feature", "importance"],
                            )
                            .properties(height=350)
                        )
                        st.altair_chart(styled(importance_chart), width="stretch")
                download_csv_button(
                    "Download node predictions", node_results, "node_predictions.csv"
                )
            except MLInputError as error:
                st.warning(str(error))
        st.markdown(
            '<p class="method-note">Scores are out-of-fold estimates. The GCN is '
            "transductive: it uses the full network structure but only training-fold labels. "
            "Included labels are synthetic; replace them with experimental outcomes.</p>",
            unsafe_allow_html=True,
        )

    with link_ml_subtab:
        section_heading(
            "Missing-link prediction",
            "Evaluate held-out edge recovery, then rank unobserved connections for follow-up.",
            "Predict",
        )
        link_model = st.radio(
            "Model",
            ["Random forest", "Graph autoencoder"],
            horizontal=True,
            key="link_model",
            captions=[
                "Baseline using structural pair features.",
                "GCN encoder with a dot-product reconstruction decoder.",
            ],
            help="Choose a link-ranking model. Both options are evaluated on held-out observed edges.",
        )
        try:
            if link_model == "Graph autoencoder":
                link_scores, link_predictions, link_diagnostic = neural_link_prediction(graph_a)
            else:
                link_scores, link_predictions, link_diagnostic = learned_link_prediction(graph_a)
            st.caption(f"Model: {link_scores.get('model', 'class-balanced random forest')}")
            metric_row(
                [
                    ("Training edges", link_scores["training_edges"], None),
                    ("Held-out edges", link_scores["held_out_edges"], None),
                    ("ROC AUC", f"{link_scores['roc_auc']:.3f}", None),
                    ("Average precision", f"{link_scores['average_precision']:.3f}", None),
                ]
            )
            left, right = st.columns([1.3, 1], gap="large")
            with left:
                st.dataframe(link_predictions, hide_index=True, width="stretch", height=390)
            with right:
                if link_model == "Graph autoencoder":
                    st.markdown("**Neural training curve**")
                    loss_chart = (
                        alt.Chart(link_diagnostic)
                        .mark_line(color="#8B5CF6", strokeWidth=3)
                        .encode(
                            x=alt.X("epoch:Q", title="Training epoch"),
                            y=alt.Y("training_loss:Q", title="Reconstruction loss"),
                            tooltip=["epoch", "training_loss"],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(styled(loss_chart), width="stretch")
                else:
                    st.markdown("**Model feature importance**")
                    st.dataframe(link_diagnostic, hide_index=True, width="stretch")
            download_csv_button(
                "Download learned link predictions",
                link_predictions,
                "learned_link_predictions.csv",
            )
        except MLInputError as error:
            st.warning(f"{error} Showing the structural ranking instead.")
            st.dataframe(top_link_predictions(graph_a, top_n=25), hide_index=True, width="stretch")
        st.markdown(
            '<p class="method-note">Evaluation hides observed edges, trains on the remaining '
            "topology, and tests recovery. Candidates still require domain validation.</p>",
            unsafe_allow_html=True,
        )

    with graph_ml_subtab:
        section_heading(
            "Graph classification",
            "Train and evaluate a model across independent networks with known graph-level labels.",
            "Predict",
        )
        graph_model = st.radio(
            "Model",
            ["Random forest", "Graph neural network"],
            horizontal=True,
            key="graph_model",
            captions=[
                "Interpretable baseline using whole-network statistics.",
                "Two-layer GCN with mean pooling across each graph.",
            ],
            help="Choose the graph classifier. Evaluation remains stratified and out of fold.",
        )
        st.caption("Upload multiple edge-list CSVs plus graph labels, or run the topology demo.")
        graph_files = st.file_uploader(
            "Graph edge lists",
            type="csv",
            accept_multiple_files=True,
            key="graph_files",
            help="Upload two or more independent edge-list CSV files with unique filename stems.",
        )
        graph_labels_file = st.file_uploader(
            "Graph labels", type="csv", help="Columns: graph, label", key="graph_labels"
        )
        try:
            if graph_files:
                names = [Path(item.name).stem for item in graph_files]
                if len(names) != len(set(names)):
                    raise MLInputError("Uploaded graph filenames must have unique stems.")
                graph_dataset = {
                    name: graph_from_edgelist(pd.read_csv(item), directed=directed)
                    for name, item in zip(names, graph_files)
                }
                if graph_labels_file is None:
                    raise MLInputError(
                        "Upload graph labels with columns named 'graph' and 'label'."
                    )
                graph_labels = pd.read_csv(graph_labels_file)
                demo_mode = False
            else:
                graph_dataset, graph_labels = demo_graph_dataset()
                demo_mode = True
            if graph_model == "Graph neural network":
                graph_scores, graph_results, graph_diagnostic = neural_graph_classification(
                    graph_dataset, graph_labels
                )
            else:
                graph_scores, graph_results, graph_diagnostic = graph_classification(
                    graph_dataset, graph_labels
                )
            st.caption(f"Model: {graph_scores.get('model', 'class-balanced random forest')}")
            metric_row(
                [
                    ("Graphs", graph_scores["graphs"], None),
                    ("CV folds", graph_scores["cv_folds"], None),
                    ("Balanced accuracy", f"{graph_scores['balanced_accuracy']:.3f}", None),
                    ("Macro F1", f"{graph_scores['macro_f1']:.3f}", None),
                ]
            )
            if demo_mode:
                st.info(
                    "Demo mode classifies synthetic modular versus hub-dominated graph families."
                )
            left, right = st.columns([1.2, 1], gap="large")
            with left:
                st.dataframe(graph_results, hide_index=True, width="stretch", height=340)
            with right:
                if graph_model == "Graph neural network":
                    graph_loss_chart = (
                        alt.Chart(graph_diagnostic)
                        .mark_line(color="#A78BFA", strokeWidth=3)
                        .encode(
                            x=alt.X("epoch:Q", title="Training epoch"),
                            y=alt.Y("training_loss:Q", title="Mean cross-validation training loss"),
                            tooltip=["epoch", "training_loss"],
                        )
                        .properties(height=340)
                    )
                    st.altair_chart(styled(graph_loss_chart), width="stretch")
                else:
                    st.markdown("**Model feature importance**")
                    st.dataframe(graph_diagnostic, hide_index=True, width="stretch", height=340)
            download_csv_button(
                "Download graph predictions", graph_results, "graph_predictions.csv"
            )
        except (MLInputError, GraphInputError, pd.errors.ParserError) as error:
            st.warning(str(error))
        st.markdown(
            '<p class="method-note">Graph classification needs independent networks as '
            "samples and at least two samples per class. Cross-validation is stratified.</p>",
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    '<div class="app-footer">NodeSafari · Open-source network discovery for research · '
    "v1.5.0 · Exploratory outputs require domain validation</div>",
    unsafe_allow_html=True,
)
