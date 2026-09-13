"""Streamlit application for NodeSafari."""

from __future__ import annotations

from html import escape
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

st.set_page_config(page_title="NodeSafari · Network Discovery", page_icon="🧭", layout="wide")
st.markdown(
    """
    <style>
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
    .block-container { max-width:1480px; padding-top:2.25rem; padding-bottom:3rem; }
    h1,h2,h3 { letter-spacing:-0.025em; }
    h1 { font-size:2.35rem !important; margin:.15rem 0 .35rem !important; }
    h2 { margin-top:.25rem !important; }
    .brand-lockup { display:flex; align-items:center; gap:.7rem; margin:.1rem 0 1.3rem; }
    .brand-mark {
      width:2.35rem; height:2.35rem; border-radius:12px; display:grid; place-items:center;
      background:linear-gradient(135deg,#7c3aed,#0f9f91); color:white; font-size:1.25rem;
      box-shadow:0 8px 18px rgba(91,33,182,.18);
    }
    .brand-name { color:#201a3b; font-size:1.08rem; font-weight:760; letter-spacing:-.02em; }
    .brand-version { color:#807a91; font-size:.72rem; font-weight:650; letter-spacing:.08em; text-transform:uppercase; }
    .hero {
      padding:1.25rem 1.4rem 1.15rem; margin-bottom:1rem; border:1px solid var(--line);
      border-radius:18px; background:rgba(255,255,255,.9);
      box-shadow:0 16px 42px rgba(54,35,93,.07);
    }
    .eyebrow { color:#0f9f91; font-size:.74rem; font-weight:750; letter-spacing:.16em; text-transform:uppercase; }
    .hero-title { color:#201a3b; font-size:2.35rem; font-weight:770; letter-spacing:-.045em; line-height:1.05; margin:.3rem 0 .45rem; }
    .lede { color:#625d76; font-size:1.02rem; max-width:900px; margin:0; line-height:1.55; }
    .workflow { display:flex; flex-wrap:wrap; gap:.42rem; margin-top:.9rem; }
    .workflow span { color:#5d5570; background:#f8f7fc; border:1px solid #e6e0f0; border-radius:999px; padding:.28rem .62rem; font-size:.76rem; font-weight:650; }
    .workflow span::after { content:"›"; color:#14b8a6; margin-left:.52rem; }
    .workflow span:last-child::after { content:""; margin:0; }
    .context-strip { display:flex; flex-wrap:wrap; align-items:center; gap:.5rem; margin:.15rem 0 1.15rem; }
    .context-label { color:#827b93; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-right:.1rem; }
    .context-chip { color:#4f4762; background:white; border:1px solid #ded8ea; border-radius:999px; padding:.26rem .62rem; font-size:.76rem; }
    .context-chip strong { color:#30294e; }
    .context-chip.demo { background:#f5f3ff; border-color:#ddd6fe; color:#6d28d9; }
    .context-chip.upload { background:#ecfdf9; border-color:#99f6e4; color:#0f766e; }
    [data-testid="stMetric"] {
      background:linear-gradient(145deg,#ffffff,#fbfaff);
      border:1px solid #e2dcf2;
      box-shadow:0 8px 24px rgba(55,35,100,.055),inset 0 3px 0 rgba(45,212,191,.65);
      padding:.88rem 1rem;
      border-radius:14px; min-height:92px;
    }
    [data-testid="stMetricLabel"] { color:#68637d; font-weight:600; }
    [data-testid="stMetricValue"] { color:#241d40; letter-spacing:-.035em; }
    .section-kicker { color:#0f9f91; font-size:.7rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.2rem; }
    .section-title { color:#261f42; font-size:1.32rem; font-weight:740; letter-spacing:-.025em; margin:0 0 .22rem; }
    .section-description { color:#716b84; font-size:.9rem; line-height:1.5; margin:0 0 1rem; max-width:900px; }
    .insight { border-left:3px solid #14b8a6; background:linear-gradient(90deg,#ecfdf9,#f7f5ff); padding:.85rem 1rem; border-radius:0 12px 12px 0; color:#30294e; line-height:1.5; }
    .empty-state { border:1px dashed #cfc5e8; background:#fbfaff; padding:1.35rem; border-radius:14px; color:#625d76; }
    .empty-state strong { color:#30294e; display:block; margin-bottom:.25rem; }
    .source-card { border:1px solid #e4deef; background:#ffffff; border-radius:12px; padding:.72rem .8rem; margin:.4rem 0 .9rem; }
    .source-card .label { color:#847d95; font-size:.68rem; font-weight:750; letter-spacing:.1em; text-transform:uppercase; }
    .source-card .value { color:#30294e; font-size:.86rem; font-weight:680; margin-top:.16rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .source-card .meta { color:#817a91; font-size:.73rem; margin-top:.12rem; }
    .method-note { color:#716b84; font-size:.84rem; }
    .stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid #e9e4f1; padding-bottom:.45rem; }
    .stTabs [data-baseweb="tab"] { background:transparent; border:1px solid transparent; border-radius:10px; padding:.5rem .84rem; color:#6b647d; font-weight:620; }
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


def styled(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(gridColor="#e8e3f2", labelColor="#625d76", titleColor="#30294e")
        .configure_legend(labelColor="#625d76", titleColor="#30294e")
        .configure_view(strokeOpacity=0)
    )


def metric_row(values: list[tuple[str, object, str | None]]) -> None:
    columns = st.columns(len(values))
    for column, (label, value, help_text) in zip(columns, values):
        column.metric(label, value, help=help_text)


def section_heading(title: str, description: str, kicker: str | None = None) -> None:
    kicker_html = f'<div class="section-kicker">{escape(kicker)}</div>' if kicker else ""
    st.markdown(
        f'{kicker_html}<div class="section-title">{escape(title)}</div>'
        f'<div class="section-description">{escape(description)}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, description: str) -> None:
    st.markdown(
        f'<div class="empty-state"><strong>{escape(title)}</strong>{escape(description)}</div>',
        unsafe_allow_html=True,
    )


def source_card(label: str, value: str, metadata: str) -> None:
    st.markdown(
        f'<div class="source-card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div><div class="meta">{escape(metadata)}</div></div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">⌁</div><div>'
        '<div class="brand-name">NodeSafari</div><div class="brand-version">Research workspace · v1.2</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Data workspace")
    st.caption("Bring an edge list, or explore the complete workflow with synthetic examples.")
    uploaded_a = st.file_uploader(
        "Reference network (A)", type="csv", help="CSV columns: source, target, optional weight"
    )
    uploaded_b = st.file_uploader(
        "Comparison network (B)", type="csv", help="Optional second network or condition"
    )
    primary_is_demo = uploaded_a is None
    comparison_is_demo = uploaded_b is None and primary_is_demo
    source_card(
        "Active reference",
        "Synthetic reference example" if primary_is_demo else uploaded_a.name,
        "Demo data" if primary_is_demo else "Uploaded CSV",
    )
    if uploaded_b is not None:
        source_card("Active comparison", uploaded_b.name, "Uploaded CSV")
    elif comparison_is_demo:
        source_card("Active comparison", "Synthetic comparison example", "Demo data")
    else:
        source_card("Active comparison", "Not loaded", "Upload B to enable comparison")

    with st.expander("Analysis settings", expanded=False):
        directed = st.toggle("Directed network", value=False)
        randomizations = st.slider("Rich-club null networks", 5, 100, 20, step=5)
        robustness_repeats = st.slider("Random robustness repeats", 10, 100, 30, step=10)
        st.caption("Higher repeat counts improve stability but take longer.")
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

st.markdown(
    '<div class="hero"><div class="eyebrow">NodeSafari · interpretable network discovery</div>'
    '<div class="hero-title">Turn connected data into testable questions.</div>'
    '<div class="lede">Explore structure, compare networks, simulate perturbations, and '
    "evaluate graph-based predictions in one reproducible workspace.</div>"
    '<div class="workflow"><span>Validate</span><span>Explore</span><span>Compare</span>'
    "<span>Perturb</span><span>Predict</span></div></div>",
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
st.markdown(
    '<div class="context-strip"><span class="context-label">Active analysis</span>'
    f'<span class="context-chip {primary_class}"><strong>A</strong> · {primary_source}</span>'
    f'<span class="context-chip {comparison_class}"><strong>B</strong> · {comparison_source}</span>'
    f'<span class="context-chip">{"Directed" if directed else "Undirected"}</span>'
    f'<span class="context-chip">{"Weighted" if "weight" in edges_a.columns else "Unweighted"}</span>'
    "</div>",
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
    ["01  Data & QC", "02  Explore", "03  Compare", "04  Perturb", "05  Predict"]
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
        st.download_button("Download A QC report", csv_bytes(qc_a), "network_a_qc.csv", "text/csv")
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
            st.download_button(
                "Download B QC report", csv_bytes(qc_b), "network_b_qc.csv", "text/csv"
            )
    st.markdown(
        '<p class="method-note">Warnings identify analyzable conditions that may change '
        "interpretation. Errors must be corrected before graph construction.</p>",
        unsafe_allow_html=True,
    )
    manifest = pd.DataFrame(
        [
            {"setting": "NodeSafari version", "value": "1.2.0"},
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
            {"setting": "Robustness repeats", "value": str(robustness_repeats)},
        ]
    )
    with st.expander("Analysis manifest", expanded=False):
        st.caption("Save the active inputs and settings alongside exported results.")
        st.dataframe(manifest, hide_index=True, width="stretch")
        st.download_button(
            "Download analysis manifest",
            csv_bytes(manifest),
            "nodesafari_analysis_manifest.csv",
            "text/csv",
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
            st.download_button(
                "Download node metrics", csv_bytes(metrics), "node_metrics.csv", "text/csv"
            )

    with rich_subtab:
        section_heading(
            "Rich-club organization",
            "Test whether high-degree nodes connect more densely than expected under a degree-preserving null model.",
            "Explore",
        )
        curve = rich_club_curve(graph_a, randomizations=randomizations)
        chart_frame = curve.dropna(subset=["normalized_phi"])
        if chart_frame.empty:
            st.info("This graph is too small or sparse for a stable normalized curve.")
        else:
            chart = (
                alt.Chart(chart_frame)
                .mark_line(point=True, color="#2DD4BF", strokeWidth=3)
                .encode(
                    x=alt.X("degree_threshold:Q", title="Degree threshold (k)"),
                    y=alt.Y("normalized_phi:Q", title="Normalized rich-club coefficient"),
                    tooltip=["degree_threshold", "observed_phi", "null_phi", "normalized_phi"],
                )
                .properties(height=360)
            )
            rule = (
                alt.Chart(pd.DataFrame({"y": [1.0]}))
                .mark_rule(color="#A78BFA", strokeDash=[5, 5])
                .encode(y="y:Q")
            )
            st.altair_chart(styled(chart + rule), width="stretch")
            peak = chart_frame.loc[chart_frame["normalized_phi"].idxmax()]
            st.markdown(
                f'<div class="insight">Peak enrichment is above degree '
                f"<b>{int(peak.degree_threshold)}</b> (normalized φ = "
                f"<b>{peak.normalized_phi:.2f}</b>). Values above 1 are denser than the "
                "degree-preserving null ensemble.</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p class="method-note">Inspect the full curve. Small and sparse graphs can '
            "produce unstable normalized estimates.</p>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download rich-club curve", csv_bytes(curve), "rich_club_curve.csv", "text/csv"
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
            st.download_button(
                "Download communities", csv_bytes(communities), "communities.csv", "text/csv"
            )
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
            st.download_button(
                "Download core positions", csv_bytes(cores), "core_periphery.csv", "text/csv"
            )
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
            st.download_button(
                "Download bridge nodes",
                csv_bytes(bridge_nodes),
                "articulation_nodes.csv",
                "text/csv",
            )
            st.download_button(
                "Download bridge edges", csv_bytes(bridge_edges), "bridge_edges.csv", "text/csv"
            )
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
        st.download_button(
            "Download network statistics",
            csv_bytes(statistics),
            "network_statistics.csv",
            "text/csv",
        )

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
            st.download_button(
                "Download differential nodes",
                csv_bytes(node_comparison),
                "differential_nodes.csv",
                "text/csv",
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
            st.download_button(
                "Download differential communities",
                csv_bytes(node_assignments),
                "differential_communities.csv",
                "text/csv",
            )
            st.markdown(
                '<p class="method-note">B communities are aligned to A by maximal node overlap '
                "before reassignment is reported.</p>",
                unsafe_allow_html=True,
            )

        with rich_compare_subtab:
            section_heading(
                "Differential rich-club organization",
                "Compare normalized rich-club profiles across matched degree thresholds.",
                "Compare",
            )
            rich_difference = differential_rich_club(
                graph_a, graph_b, randomizations=randomizations
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
                        x=alt.X("degree_threshold:Q", title="Degree threshold (k)"),
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
            st.download_button(
                "Download differential rich club",
                csv_bytes(rich_difference),
                "differential_rich_club.csv",
                "text/csv",
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
        st.download_button(
            "Download node screen",
            csv_bytes(perturbations),
            "node_perturbation_screen.csv",
            "text/csv",
        )

    with edge_perturb_subtab:
        edge_perturbations = edge_perturbation_screen(graph_a)
        section_heading(
            "Single-edge deletion screen",
            "Find connections whose loss most changes global efficiency or fragments the network.",
            "Perturb",
        )
        st.dataframe(edge_perturbations, hide_index=True, width="stretch", height=430)
        st.download_button(
            "Download edge screen",
            csv_bytes(edge_perturbations),
            "edge_perturbation_screen.csv",
            "text/csv",
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
        st.download_button(
            "Download robustness curve", csv_bytes(robustness), "robustness_curve.csv", "text/csv"
        )

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
                "Find nodes with a similar network role", list(embeddings["node"])
            )
            st.dataframe(
                nearest_nodes(embeddings, query_node), hide_index=True, width="stretch", height=390
            )
        st.download_button(
            "Download embeddings", csv_bytes(embeddings), "node_embeddings.csv", "text/csv"
        )

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
                st.download_button(
                    "Download node predictions",
                    csv_bytes(node_results),
                    "node_predictions.csv",
                    "text/csv",
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
            st.download_button(
                "Download learned link predictions",
                csv_bytes(link_predictions),
                "learned_link_predictions.csv",
                "text/csv",
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
        )
        st.caption("Upload multiple edge-list CSVs plus graph labels, or run the topology demo.")
        graph_files = st.file_uploader(
            "Graph edge lists", type="csv", accept_multiple_files=True, key="graph_files"
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
            st.download_button(
                "Download graph predictions",
                csv_bytes(graph_results),
                "graph_predictions.csv",
                "text/csv",
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
    "v1.2.0 · Exploratory outputs require domain validation</div>",
    unsafe_allow_html=True,
)
