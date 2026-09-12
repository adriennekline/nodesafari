"""Streamlit application for NodeSafari."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from nodesafari.analysis import (
    community_table,
    compare_networks,
    nearest_nodes,
    network_summary,
    node_metrics,
    perturbation_screen,
    rich_club_curve,
    spectral_embedding,
    top_link_predictions,
)
from nodesafari.io import GraphInputError, graph_from_edgelist
from nodesafari.visualization import network_figure

ROOT = Path(__file__).parent
EXAMPLES = ROOT / "examples"

st.set_page_config(page_title="NodeSafari", page_icon="🧭", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink:#07121f; --panel:#0d1c2c; --teal:#18b6a4; --muted:#92a7ba; }
    .stApp { background: linear-gradient(145deg, #06101c 0%, #0b1b2a 55%, #102636 100%); color:#edf6f8; }
    [data-testid="stSidebar"] { background:#071522; border-right:1px solid #1e4051; }
    h1,h2,h3 { letter-spacing:-0.025em; }
    h1 { font-size:2.15rem !important; margin-bottom:.1rem !important; }
    .eyebrow { color:#55d7c7; font-size:.78rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .lede { color:#afc1cf; font-size:1rem; max-width:760px; margin-bottom:1rem; }
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


with st.sidebar:
    st.markdown("### Data workspace")
    st.caption("Upload an edge list or begin with the included gene-interaction example.")
    uploaded_a = st.file_uploader("Primary network", type="csv", help="CSV columns: source, target, optional weight")
    uploaded_b = st.file_uploader("Comparison network", type="csv", help="Optional second condition")
    directed = st.toggle("Directed network", value=False)
    randomizations = st.slider("Rich-club null networks", 5, 100, 20, step=5)
    st.divider()
    st.markdown("**Expected format**")
    st.code("source,target,weight\nTP53,MDM2,1.0", language="text")
    st.caption("Inputs stay in the current Streamlit session. Demo data are synthetic and illustrative.")


try:
    edges_a = read_csv(uploaded_a, EXAMPLES / "control_edges.csv")
    edges_b = read_csv(uploaded_b, EXAMPLES / "disease_edges.csv")
    graph_a = graph_from_edgelist(edges_a, directed=directed)
    graph_b = graph_from_edgelist(edges_b, directed=directed)
except (GraphInputError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()

st.markdown('<div class="eyebrow">Biological network discovery</div>', unsafe_allow_html=True)
st.title("NodeSafari")
st.markdown(
    '<div class="lede">Move from network structure to testable hypotheses: discover modules, compare conditions, simulate perturbations, and learn graph-based representations.</div>',
    unsafe_allow_html=True,
)

summary = network_summary(graph_a)
metric_columns = st.columns(5)
for column, label, key, fmt in zip(
    metric_columns,
    ["Nodes", "Interactions", "Modules", "Density", "Connected"],
    ["nodes", "edges", None, "density", "largest_component_fraction"],
    ["{:.0f}", "{:.0f}", "{:.0f}", "{:.3f}", "{:.0%}"],
):
    value = len(community_table(graph_a)["community"].unique()) if key is None else summary[key]
    column.metric(label, fmt.format(value))

overview_tab, structure_tab, compare_tab, perturb_tab, ml_tab = st.tabs(
    ["Explore", "Rich club", "Compare", "Perturb", "ML lab"]
)

communities = community_table(graph_a)
metrics = node_metrics(graph_a)

with overview_tab:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.pyplot(network_figure(graph_a, communities), width="stretch")
    with right:
        st.subheader("High-interest nodes")
        display = metrics[["node", "degree", "betweenness", "pagerank"]].head(10)
        st.dataframe(display, hide_index=True, width="stretch")
        top_node = str(display.iloc[0]["node"])
        st.markdown(
            f'<div class="insight"><b>{top_node}</b> is the leading hub in this network. Treat this as a structural candidate for follow-up, not evidence of biological causality.</div>',
            unsafe_allow_html=True,
        )
        st.download_button("Download node metrics", csv_bytes(metrics), "node_metrics.csv", "text/csv")

with structure_tab:
    st.subheader("Rich-club organization")
    curve = rich_club_curve(graph_a, randomizations=randomizations)
    chart_frame = curve.dropna(subset=["normalized_phi"])
    if chart_frame.empty:
        st.info("This network is too small or sparse for a stable normalized rich-club curve.")
    else:
        chart = (
            alt.Chart(chart_frame)
            .mark_line(point=True, color="#18B6A4", strokeWidth=3)
            .encode(
                x=alt.X("degree_threshold:Q", title="Degree threshold (k)"),
                y=alt.Y("normalized_phi:Q", title="Normalized rich-club coefficient"),
                tooltip=["degree_threshold", alt.Tooltip("observed_phi", format=".3f"), alt.Tooltip("null_phi", format=".3f"), alt.Tooltip("normalized_phi", format=".3f")],
            )
            .properties(height=360)
        )
        rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="#F59E5B", strokeDash=[5, 5]).encode(y="y:Q")
        st.altair_chart((chart + rule).configure_axis(gridColor="#284456", labelColor="#BDD0DB", titleColor="#E5F0F3").configure_view(strokeOpacity=0), width="stretch")
        peak = chart_frame.loc[chart_frame["normalized_phi"].idxmax()]
        st.markdown(
            f'<div class="insight">The strongest enrichment occurs above degree <b>{int(peak.degree_threshold)}</b> (normalized φ = <b>{peak.normalized_phi:.2f}</b>). Values above 1 indicate denser connectivity than degree-preserving null networks.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<p class="method-note">Normalized against degree-preserving randomized networks. Results can be unstable for small or very sparse graphs; inspect the full curve rather than a single threshold.</p>', unsafe_allow_html=True)
    st.download_button("Download rich-club curve", csv_bytes(curve), "rich_club_curve.csv", "text/csv")

with compare_tab:
    st.subheader("Condition A → Condition B")
    global_comparison, node_comparison = compare_networks(graph_a, graph_b)
    first, second = st.columns([.9, 1.35], gap="large")
    with first:
        st.markdown("**Whole-network changes**")
        st.dataframe(global_comparison, hide_index=True, width="stretch")
    with second:
        st.markdown("**Nodes with the largest degree shifts**")
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
        st.altair_chart(shift_chart.configure_axis(gridColor="#284456", labelColor="#BDD0DB", titleColor="#E5F0F3").configure_view(strokeOpacity=0), width="stretch")
    st.caption("The included example treats control as A and disease-like rewiring as B. Uploaded comparison data replace B.")
    st.download_button("Download differential nodes", csv_bytes(node_comparison), "differential_nodes.csv", "text/csv")

with perturb_tab:
    st.subheader("In-silico single-node deletion screen")
    perturbations = perturbation_screen(graph_a)
    chart = (
        alt.Chart(perturbations.head(15))
        .mark_bar(color="#F59E5B", cornerRadiusEnd=4)
        .encode(
            x=alt.X("impact_score:Q", title="Structural impact score"),
            y=alt.Y("node:N", sort="-x", title=None),
            tooltip=["node", alt.Tooltip("impact_score", format=".4f"), alt.Tooltip("efficiency_change", format=".4f"), "components_after"],
        )
        .properties(height=420)
    )
    st.altair_chart(chart.configure_axis(gridColor="#284456", labelColor="#BDD0DB", titleColor="#E5F0F3").configure_view(strokeOpacity=0), width="stretch")
    if not perturbations.empty:
        lead = perturbations.iloc[0]
        st.markdown(
            f'<div class="insight">Removing <b>{lead.node}</b> produces the largest structural impact in this screen. This ranking combines global-efficiency loss and fragmentation; it is hypothesis-generating, not a molecular intervention model.</div>',
            unsafe_allow_html=True,
        )
    st.download_button("Download perturbation screen", csv_bytes(perturbations), "perturbation_screen.csv", "text/csv")

with ml_tab:
    embeddings = spectral_embedding(graph_a, dimensions=8)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Learned node map")
        if {"embedding_1", "embedding_2"}.issubset(embeddings.columns):
            plotted = embeddings.merge(communities, on="node", how="left")
            scatter = (
                alt.Chart(plotted)
                .mark_circle(size=115, opacity=.86)
                .encode(
                    x=alt.X("embedding_1:Q", title="Embedding dimension 1"),
                    y=alt.Y("embedding_2:Q", title="Embedding dimension 2"),
                    color=alt.Color("community:N", legend=alt.Legend(title="Module")),
                    tooltip=["node", "community"],
                )
                .properties(height=390)
            )
            st.altair_chart(scatter.configure_axis(gridColor="#284456", labelColor="#BDD0DB", titleColor="#E5F0F3").configure_view(strokeOpacity=0), width="stretch")
        query_node = st.selectbox("Find nodes with a similar network role", list(embeddings["node"]))
        st.dataframe(nearest_nodes(embeddings, query_node), hide_index=True, width="stretch")
    with right:
        st.subheader("Candidate missing interactions")
        predictions = top_link_predictions(graph_a, top_n=20)
        st.dataframe(predictions, hide_index=True, width="stretch", height=390)
        st.markdown('<p class="method-note">Predictions combine common neighbors, Jaccard similarity, and Adamic–Adar. They prioritize structurally plausible absent edges and require experimental validation.</p>', unsafe_allow_html=True)
        st.download_button("Download link predictions", csv_bytes(predictions), "link_predictions.csv", "text/csv")

st.divider()
st.caption("NodeSafari · Open-source network discovery for basic science · v0.1.0")
