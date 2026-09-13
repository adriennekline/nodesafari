"""Differential analyses for two networks."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .analysis import community_table, compare_networks, rich_club_curve


def differential_community_analysis(
    graph_a, graph_b
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Compare community organization and align B labels to A by overlap."""

    table_a = community_table(graph_a).rename(columns={"community": "community_a"})
    table_b = community_table(graph_b).rename(columns={"community": "community_b"})
    common = table_a.merge(table_b, on="node", how="inner")
    if common.empty:
        summary = {
            "common_nodes": 0,
            "adjusted_rand_index": float("nan"),
            "normalized_mutual_information": float("nan"),
            "fraction_reassigned": float("nan"),
        }
        return summary, common, pd.DataFrame()

    overlaps = (
        common.groupby(["community_b", "community_a"])
        .size()
        .reset_index(name="overlap")
        .sort_values("overlap", ascending=False)
    )
    mapping: dict[int, int] = {}
    used_a: set[int] = set()
    for row in overlaps.itertuples(index=False):
        if row.community_b not in mapping and row.community_a not in used_a:
            mapping[int(row.community_b)] = int(row.community_a)
            used_a.add(int(row.community_a))
    next_label = max(table_a["community_a"].max(), 0) + 1
    for label in sorted(table_b["community_b"].unique()):
        if int(label) not in mapping:
            mapping[int(label)] = int(next_label)
            next_label += 1

    common["community_b_aligned"] = common["community_b"].map(mapping)
    common["reassigned"] = common["community_a"] != common["community_b_aligned"]
    flows = (
        common.groupby(["community_a", "community_b_aligned"])
        .size()
        .reset_index(name="nodes")
        .sort_values("nodes", ascending=False)
    )
    summary = {
        "common_nodes": len(common),
        "adjusted_rand_index": float(
            adjusted_rand_score(common["community_a"], common["community_b"])
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(common["community_a"], common["community_b"])
        ),
        "fraction_reassigned": float(common["reassigned"].mean()),
    }
    return (
        summary,
        common.sort_values(["reassigned", "node"], ascending=[False, True]).reset_index(drop=True),
        flows.reset_index(drop=True),
    )


def differential_rich_club(
    graph_a,
    graph_b,
    *,
    randomizations: int = 20,
    seed: int = 42,
    swaps_per_edge: int = 10,
    min_rich_nodes: int = 5,
    richness: str = "degree",
    weighted: bool = False,
) -> pd.DataFrame:
    """Compare normalized rich-club curves on shared degree thresholds."""

    settings = {
        "randomizations": randomizations,
        "swaps_per_edge": swaps_per_edge,
        "min_rich_nodes": min_rich_nodes,
        "richness": richness,
        "weighted": weighted,
    }
    curve_a = rich_club_curve(graph_a, seed=seed, **settings).add_suffix("_a")
    curve_b = rich_club_curve(graph_b, seed=seed + 1, **settings).add_suffix("_b")
    merged = curve_a.merge(
        curve_b,
        left_on="degree_threshold_a",
        right_on="degree_threshold_b",
        how="outer",
    )
    merged["degree_threshold"] = merged["degree_threshold_a"].fillna(merged["degree_threshold_b"])
    merged["normalized_change"] = merged["normalized_phi_b"] - merged["normalized_phi_a"]
    columns = [
        "degree_threshold",
        "normalized_phi_a",
        "normalized_phi_b",
        "normalized_change",
        "observed_phi_a",
        "observed_phi_b",
        "null_phi_a",
        "null_phi_b",
        "p_empirical_a",
        "p_empirical_b",
        "reliable_node_count_a",
        "reliable_node_count_b",
    ]
    return merged[columns].sort_values("degree_threshold").reset_index(drop=True)


__all__ = ["compare_networks", "differential_community_analysis", "differential_rich_club"]
