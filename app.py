"""Streamlit application for NodeSafari."""

from __future__ import annotations

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
from nodesafari.visualization import network_figure

ROOT = Path(__file__).parent
EXAMPLES = ROOT / "examples"

st.set_page_config(page_title="NodeSafari", page_icon="🧭", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink:#07121f; --panel:#0d1c2c; --teal:#18b6a4; --muted:#92a7ba; }
    .stApp { background:linear-gradient(145deg,#06101c 0%,#0b1b2a 55%,#102636 100%); color:#edf6f8; }
    [data-testid="stSidebar"] { background:#071522; border-right:1px solid #1e4051; }
    h1,h2,h3 { letter-spacing:-0.025em; }
    h1 { font-size:2.15rem !important; margin-bottom:.1rem !important; }
    .eyebrow { color:#55d7c7; font-size:.78rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .lede { color:#afc1cf; font-size:1rem; max-width:830px; margin-bottom:1rem; }
    [data-testid="stMetric"] { background:rgba(13,28,44,.82); border:1px solid #214255; padding:.8rem 1rem; border-radius:14px; }
    [data-testid="stMetricLabel"] { color:#9eb2c1; }
    .insight { border-left:3px solid #18b6a4; background:rgba(13,28,44,.82); padding:.8rem 1rem; border-radius:0 12px 12px 0; color:#dcebee; }
    .method-note { color:#8fa6b7; font-size:.84rem; }
    .stTabs [data-baseweb="tab-list"] { gap:.35rem; }
    .stTabs [data-baseweb="tab"] { background:#0d1c2c; border-radius:10px; padding:.45rem .8rem; }
    .stTabs [aria-selected="true"] { background:#15384a; color:#75e5d6; }
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


def styled(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            gridColor="#284456", labelColor="#BDD0DB", titleColor="#E5F0F3"
        )
        .configure_legend(labelColor="#BDD0DB", titleColor="#E5F0F3")
        .configure_view(strokeOpacity=0)
    )


def metric_row(values: list[tuple[str, object, str | None]]) -> None:
    columns = st.columns(len(values))
    for column, (label, value, help_text) in zip(columns, values):
        column.metric(label, value, help=help_text)


with st.sidebar:
    st.markdown("### Data workspace")
    st.caption("Upload edge lists or begin with the included synthetic examples.")
    uploaded_a = st.file_uploader(
        "Primary network (A)", type="csv", help="CSV columns: source, target, optional weight"
    )
    uploaded_b = st.file_uploader(
        "Comparison network (B)", type="csv", help="Optional second condition"
    )
    directed = st.toggle("Directed network", value=False)
    randomizations = st.slider("Rich-club null networks", 5, 100, 20, step=5)
    robustness_repeats = st.slider("Random robustness repeats", 10, 100, 30, step=10)
    st.divider()
    st.markdown("**Expected edge-list format**")
    st.code("source,target,weight\nTP53,MDM2,1.0", language="text")
    st.caption("Inputs stay in this Streamlit session. Demo data are synthetic and illustrative.")


try:
    edges_a = read_csv(uploaded_a, EXAMPLES / "control_edges.csv")
    edges_b = read_csv(uploaded_b, EXAMPLES / "disease_edges.csv")
    qc_a = network_qc(edges_a, directed=directed)
    qc_b = network_qc(edges_b, directed=directed)
    graph_a = graph_from_edgelist(edges_a, directed=directed)
    graph_b = graph_from_edgelist(edges_b, directed=directed)
except (GraphInputError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(str(error))
    st.stop()

st.markdown('<div class="eyebrow">Biological network discovery</div>', unsafe_allow_html=True)
st.title("NodeSafari")
st.markdown(
    '<div class="lede">Upload → quality-check → explore → compare → perturb → predict. '
    "Every result stays exportable and every model reports validation metrics.</div>",
    unsafe_allow_html=True,
)

summary = network_summary(graph_a)
communities = community_table(graph_a)
metrics = node_metrics(graph_a)
metric_row(
    [
        ("Nodes", f"{summary['nodes']:.0f}", None),
        ("Interactions", f"{summary['edges']:.0f}", None),
        ("Modules", f"{communities['community'].nunique():.0f}", None),
        ("Density", f"{summary['density']:.3f}", None),
        ("Largest component", f"{summary['largest_component_fraction']:.0%}", None),
    ]
)

qc_tab, explore_tab, compare_tab, perturb_tab, ml_tab = st.tabs(
    ["Data & QC", "Explore", "Compare", "Perturb", "ML"]
)

with qc_tab:
    st.subheader("Input quality control")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Network A**")
        st.dataframe(qc_a.assign(value=qc_a["value"].astype(str)), hide_index=True, width="stretch")
        st.download_button("Download A QC report", csv_bytes(qc_a), "network_a_qc.csv", "text/csv")
    with right:
        st.markdown("**Network B**")
        st.dataframe(qc_b.assign(value=qc_b["value"].astype(str)), hide_index=True, width="stretch")
        st.download_button("Download B QC report", csv_bytes(qc_b), "network_b_qc.csv", "text/csv")
    st.markdown(
        '<p class="method-note">Warnings identify analyzable conditions that may change '
        "interpretation. Errors must be corrected before graph construction.</p>",
        unsafe_allow_html=True,
    )

with explore_tab:
    overview_subtab, rich_subtab, community_subtab, core_subtab, stats_subtab = st.tabs(
        ["Overview & hubs", "Rich club", "Communities", "Core & bridges", "Statistics"]
    )

    with overview_subtab:
        left, right = st.columns([1.45, 1], gap="large")
        with left:
            st.pyplot(network_figure(graph_a, communities), width="stretch")
        with right:
            st.subheader("High-interest nodes")
            display = metrics[["node", "degree", "betweenness", "closeness", "pagerank"]].head(12)
            st.dataframe(display, hide_index=True, width="stretch")
            top_node = str(display.iloc[0]["node"])
            st.markdown(
                f'<div class="insight"><b>{top_node}</b> is the leading degree hub. '
                "Centrality is a structural signal, not evidence of causality.</div>",
                unsafe_allow_html=True,
            )
            st.download_button("Download node metrics", csv_bytes(metrics), "node_metrics.csv", "text/csv")

    with rich_subtab:
        st.subheader("Rich-club organization")
        curve = rich_club_curve(graph_a, randomizations=randomizations)
        chart_frame = curve.dropna(subset=["normalized_phi"])
        if chart_frame.empty:
            st.info("This graph is too small or sparse for a stable normalized curve.")
        else:
            chart = (
                alt.Chart(chart_frame)
                .mark_line(point=True, color="#18B6A4", strokeWidth=3)
                .encode(
                    x=alt.X("degree_threshold:Q", title="Degree threshold (k)"),
                    y=alt.Y("normalized_phi:Q", title="Normalized rich-club coefficient"),
                    tooltip=["degree_threshold", "observed_phi", "null_phi", "normalized_phi"],
                )
                .properties(height=360)
            )
            rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
                color="#F59E5B", strokeDash=[5, 5]
            ).encode(y="y:Q")
            st.altair_chart(styled(chart + rule), width="stretch")
            peak = chart_frame.loc[chart_frame["normalized_phi"].idxmax()]
            st.markdown(
                f'<div class="insight">Peak enrichment is above degree '
                f'<b>{int(peak.degree_threshold)}</b> (normalized φ = '
                f'<b>{peak.normalized_phi:.2f}</b>). Values above 1 are denser than the '
                "degree-preserving null ensemble.</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p class="method-note">Inspect the full curve. Small and sparse graphs can '
            "produce unstable normalized estimates.</p>",
            unsafe_allow_html=True,
        )
        st.download_button("Download rich-club curve", csv_bytes(curve), "rich_club_curve.csv", "text/csv")

    with community_subtab:
        st.subheader("Detected communities")
        community_sizes = communities.groupby("community").size().reset_index(name="nodes")
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            community_chart = (
                alt.Chart(community_sizes)
                .mark_bar(color="#18B6A4", cornerRadiusEnd=4)
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
            st.download_button("Download communities", csv_bytes(communities), "communities.csv", "text/csv")
        st.markdown(
            '<p class="method-note">Communities use greedy modularity optimization and '
            "should be interpreted alongside biological annotation.</p>",
            unsafe_allow_html=True,
        )

    with core_subtab:
        cores = core_periphery_table(graph_a)
        bridge_nodes, bridge_edges = bridge_analysis(graph_a)
        st.subheader("K-core and bridge structure")
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.markdown("**Core–periphery position**")
            st.dataframe(cores, hide_index=True, width="stretch", height=380)
            st.download_button("Download core positions", csv_bytes(cores), "core_periphery.csv", "text/csv")
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
            st.download_button("Download bridge nodes", csv_bytes(bridge_nodes), "articulation_nodes.csv", "text/csv")
            st.download_button("Download bridge edges", csv_bytes(bridge_edges), "bridge_edges.csv", "text/csv")
        st.markdown(
            '<p class="method-note">Core position uses k-core decomposition. Bridges and '
            "articulation nodes are exact disconnection points in the undirected projection.</p>",
            unsafe_allow_html=True,
        )

    with stats_subtab:
        statistics = network_statistics_table(graph_a)
        st.subheader("Whole-network statistics")
        st.dataframe(statistics, hide_index=True, width="stretch")
        st.download_button("Download network statistics", csv_bytes(statistics), "network_statistics.csv", "text/csv")

with compare_tab:
    global_subtab, community_compare_subtab, rich_compare_subtab = st.tabs(
        ["Network & hubs", "Communities", "Rich club"]
    )

    with global_subtab:
        st.subheader("Condition A → condition B")
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
                    color=alt.condition(alt.datum.degree_change >= 0, alt.value("#18B6A4"), alt.value("#EE6C91")),
                    tooltip=["node", "degree_a", "degree_b", "degree_change"],
                )
                .properties(height=390)
            )
            st.altair_chart(styled(shift_chart), width="stretch")
        st.download_button("Download differential nodes", csv_bytes(node_comparison), "differential_nodes.csv", "text/csv")

    with community_compare_subtab:
        community_summary, node_assignments, community_flows = differential_community_analysis(graph_a, graph_b)
        metric_row(
            [
                ("Common nodes", community_summary["common_nodes"], None),
                ("Adjusted Rand", f"{community_summary['adjusted_rand_index']:.3f}", "1 means identical partitions."),
                ("Normalized mutual info", f"{community_summary['normalized_mutual_information']:.3f}", "1 means identical partitions."),
                ("Reassigned", f"{community_summary['fraction_reassigned']:.0%}", "After overlap alignment."),
            ]
        )
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.markdown("**Node-level assignments**")
            st.dataframe(node_assignments, hide_index=True, width="stretch", height=360)
        with right:
            st.markdown("**Community flows**")
            st.dataframe(community_flows, hide_index=True, width="stretch", height=360)
        st.download_button("Download differential communities", csv_bytes(node_assignments), "differential_communities.csv", "text/csv")
        st.markdown(
            '<p class="method-note">B communities are aligned to A by maximal node overlap '
            "before reassignment is reported.</p>",
            unsafe_allow_html=True,
        )

    with rich_compare_subtab:
        st.subheader("Differential rich-club organization")
        rich_difference = differential_rich_club(graph_a, graph_b, randomizations=randomizations)
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
                    x=alt.X("degree_threshold:Q", title="Degree threshold (k)"),
                    y=alt.Y("normalized_phi:Q", title="Normalized rich-club coefficient"),
                    color=alt.Color("network:N", scale=alt.Scale(range=["#18B6A4", "#EE6C91"]), title=None),
                    tooltip=["network", "degree_threshold", "normalized_phi"],
                )
                .properties(height=360)
            )
            st.altair_chart(styled(rich_chart), width="stretch")
        st.dataframe(rich_difference, hide_index=True, width="stretch", height=260)
        st.download_button("Download differential rich club", csv_bytes(rich_difference), "differential_rich_club.csv", "text/csv")

with perturb_tab:
    node_perturb_subtab, edge_perturb_subtab, robustness_subtab = st.tabs(
        ["Remove nodes", "Remove edges", "Robustness"]
    )

    with node_perturb_subtab:
        perturbations = perturbation_screen(graph_a)
        st.subheader("Single-node deletion screen")
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            perturb_chart = (
                alt.Chart(perturbations.head(15))
                .mark_bar(color="#F59E5B", cornerRadiusEnd=4)
                .encode(
                    x=alt.X("impact_score:Q", title="Structural impact score"),
                    y=alt.Y("node:N", sort="-x", title=None),
                    tooltip=["node", "impact_score", "efficiency_change", "components_after"],
                )
                .properties(height=420)
            )
            st.altair_chart(styled(perturb_chart), width="stretch")
        with right:
            selected_node = st.selectbox("Inspect a node deletion", perturbations["node"])
            selected_result = perturbations.loc[perturbations["node"] == selected_node].iloc[0]
            metric_row(
                [
                    ("Impact", f"{selected_result['impact_score']:.4f}", None),
                    ("Efficiency Δ", f"{selected_result['efficiency_change']:.4f}", None),
                    ("Components", int(selected_result["components_after"]), None),
                ]
            )
            st.dataframe(perturbations, hide_index=True, width="stretch", height=310)
        st.download_button("Download node screen", csv_bytes(perturbations), "node_perturbation_screen.csv", "text/csv")

    with edge_perturb_subtab:
        edge_perturbations = edge_perturbation_screen(graph_a)
        st.subheader("Single-edge deletion screen")
        st.dataframe(edge_perturbations, hide_index=True, width="stretch", height=430)
        st.download_button("Download edge screen", csv_bytes(edge_perturbations), "edge_perturbation_screen.csv", "text/csv")
        st.markdown(
            '<p class="method-note">Impact combines global-efficiency loss and loss of the '
            "largest component. Bridge status is reported separately.</p>",
            unsafe_allow_html=True,
        )

    with robustness_subtab:
        st.subheader("Network robustness")
        robustness = robustness_curve(graph_a, random_repeats=robustness_repeats)
        plot = robustness.copy()
        plot["lower"] = (plot["largest_component_mean"] - plot["largest_component_std"]).clip(0)
        plot["upper"] = (plot["largest_component_mean"] + plot["largest_component_std"]).clip(upper=1)
        random_plot = plot[plot["strategy"] == "random failure"]
        band = alt.Chart(random_plot).mark_area(color="#6F9DB5", opacity=0.2).encode(
            x="fraction_removed:Q", y="lower:Q", y2="upper:Q"
        )
        lines = (
            alt.Chart(plot)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("fraction_removed:Q", title="Fraction of nodes removed", axis=alt.Axis(format="%")),
                y=alt.Y("largest_component_mean:Q", title="Largest component / original nodes"),
                color=alt.Color("strategy:N", scale=alt.Scale(range=["#6F9DB5", "#EE6C91"]), title=None),
                tooltip=["strategy", "fraction_removed", "largest_component_mean"],
            )
            .properties(height=390)
        )
        st.altair_chart(styled(band + lines), width="stretch")
        st.download_button("Download robustness curve", csv_bytes(robustness), "robustness_curve.csv", "text/csv")

with ml_tab:
    embedding_subtab, node_ml_subtab, link_ml_subtab, graph_ml_subtab = st.tabs(
        ["Node embeddings", "Node prediction", "Link prediction", "Graph classification"]
    )

    with embedding_subtab:
        embeddings = spectral_embedding(graph_a, dimensions=8)
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.subheader("Spectral node map")
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
            query_node = st.selectbox("Find nodes with a similar network role", list(embeddings["node"]))
            st.dataframe(nearest_nodes(embeddings, query_node), hide_index=True, width="stretch", height=390)
        st.download_button("Download embeddings", csv_bytes(embeddings), "node_embeddings.csv", "text/csv")

    with node_ml_subtab:
        st.subheader("Cross-validated node-label prediction")
        node_model = st.radio(
            "Model",
            ["Random forest", "Graph neural network"],
            horizontal=True,
            key="node_model",
        )
        node_label_upload = st.file_uploader(
            "Node labels", type="csv", help="Required: node, label. Numeric columns become features.", key="node_labels"
        )
        node_labels = read_csv(node_label_upload, EXAMPLES / "node_labels.csv")
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
                    ("Balanced accuracy", f"{node_scores['balanced_accuracy']:.3f}", None),
                    ("Macro F1", f"{node_scores['macro_f1']:.3f}", None),
                ]
            )
            left, right = st.columns([1.25, 1], gap="large")
            with left:
                st.dataframe(node_results, hide_index=True, width="stretch", height=350)
            with right:
                if node_model == "Graph neural network":
                    loss_chart = (
                        alt.Chart(node_diagnostic)
                        .mark_line(color="#18B6A4", strokeWidth=3)
                        .encode(
                            x=alt.X("epoch:Q", title="Training epoch"),
                            y=alt.Y("training_loss:Q", title="Mean cross-validation training loss"),
                            tooltip=["epoch", "training_loss"],
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(styled(loss_chart), width="stretch")
                else:
                    importance_chart = (
                        alt.Chart(node_diagnostic.head(12))
                        .mark_bar(color="#18B6A4", cornerRadiusEnd=4)
                        .encode(
                            x=alt.X("importance:Q", title="Random-forest importance"),
                            y=alt.Y("feature:N", sort="-x", title=None),
                            tooltip=["feature", "importance"],
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(styled(importance_chart), width="stretch")
            st.download_button("Download node predictions", csv_bytes(node_results), "node_predictions.csv", "text/csv")
        except MLInputError as error:
            st.warning(str(error))
        st.markdown(
            '<p class="method-note">Scores are out-of-fold estimates. The GCN is '
            "transductive: it uses the full network structure but only training-fold labels. "
            "Included labels are synthetic; replace them with experimental outcomes.</p>",
            unsafe_allow_html=True,
        )

    with link_ml_subtab:
        st.subheader("Learned missing-interaction ranking")
        link_model = st.radio(
            "Model",
            ["Random forest", "Graph autoencoder"],
            horizontal=True,
            key="link_model",
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
                        .mark_line(color="#EE6C91", strokeWidth=3)
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
            st.download_button("Download learned link predictions", csv_bytes(link_predictions), "learned_link_predictions.csv", "text/csv")
        except MLInputError as error:
            st.warning(f"{error} Showing the structural ranking instead.")
            st.dataframe(top_link_predictions(graph_a, top_n=25), hide_index=True, width="stretch")
        st.markdown(
            '<p class="method-note">Evaluation hides observed edges, trains on the remaining '
            "topology, and tests recovery. Candidates still require biological validation.</p>",
            unsafe_allow_html=True,
        )

    with graph_ml_subtab:
        st.subheader("Cross-validated graph classification")
        graph_model = st.radio(
            "Model",
            ["Random forest", "Graph neural network"],
            horizontal=True,
            key="graph_model",
        )
        st.caption("Upload multiple edge-list CSVs plus graph labels, or run the topology demo.")
        graph_files = st.file_uploader("Graph edge lists", type="csv", accept_multiple_files=True, key="graph_files")
        graph_labels_file = st.file_uploader("Graph labels", type="csv", help="Columns: graph, label", key="graph_labels")
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
                    raise MLInputError("Upload graph labels with columns named 'graph' and 'label'.")
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
                st.info("Demo mode classifies synthetic modular versus hub-dominated graph families.")
            left, right = st.columns([1.2, 1], gap="large")
            with left:
                st.dataframe(graph_results, hide_index=True, width="stretch", height=340)
            with right:
                if graph_model == "Graph neural network":
                    graph_loss_chart = (
                        alt.Chart(graph_diagnostic)
                        .mark_line(color="#F59E5B", strokeWidth=3)
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
            st.download_button("Download graph predictions", csv_bytes(graph_results), "graph_predictions.csv", "text/csv")
        except (MLInputError, GraphInputError, pd.errors.ParserError) as error:
            st.warning(str(error))
        st.markdown(
            '<p class="method-note">Graph classification needs independent networks as '
            "samples and at least two samples per class. Cross-validation is stratified.</p>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption("NodeSafari · Open-source network discovery for basic science · v1.1.0")
