# Methods and interpretation

NodeSafari is designed for exploratory and hypothesis-generating analysis. It does not infer molecular causality.

## Rich-club analysis

For degree threshold `k`, the rich-club coefficient is the density of the subgraph induced by nodes with degree greater than `k`. The normalized coefficient divides the observed density by the mean density from degree-preserving randomized networks. Values greater than one suggest enrichment relative to the selected null model.

## Communities

Communities are estimated with greedy modularity maximization. Community labels are arbitrary identifiers and should not be interpreted as biological classes without external validation.

For two-network comparison, NodeSafari calculates adjusted Rand index and normalized mutual information on shared nodes. Community labels in network B are greedily aligned to network A by node overlap before the fraction of reassigned nodes is reported. This alignment aids interpretation; it is not a statistical test across biological replicates.

## Core–periphery and bridges

Core position is calculated with k-core decomposition after removing self-loops. The normalized core score divides each node's core number by the maximum observed core number. This is a discrete structural decomposition, not a fitted continuous core–periphery block model.

Articulation nodes and bridge edges are exact disconnection points in the undirected projection. Removal consequences are reported using connected-component counts and sizes.

## Perturbation screen

Each node or edge is removed once. The structural impact score combines loss of global efficiency (60%) and loss of the largest connected component (40%). This is a transparent prioritization heuristic, not a simulation of knockout biology.

Robustness analysis compares adaptive removal of the current highest-degree node with repeated random node removal. The random curve reports its mean and standard deviation. Largest-component size is normalized to the original node count.

## Spectral embeddings

Node vectors are learned by truncated singular-value decomposition of the symmetrically normalized weighted adjacency matrix. Nearby nodes have similar structural contexts; similarity does not necessarily imply shared molecular function.

## Node prediction

Node prediction combines centrality, clustering, spectral embeddings, and any uploaded numeric covariates. A class-balanced random forest is evaluated with stratified k-fold cross-validation, limited by the smallest class. The displayed predictions are out-of-fold. Accuracy, balanced accuracy, and macro F1 are reported.

## Link prediction

Observed non-bridge edges are held out, and a class-balanced random forest learns from common neighbors, Jaccard similarity, Adamic–Adar, preferential attachment, and endpoint degrees in the remaining graph. Recovery is evaluated with ROC AUC and average precision. The fitted model then ranks genuinely absent interactions. Small networks fall back to a structural ensemble. All candidates require independent experimental evidence.

## Graph classification

Each uploaded graph becomes one sample described by size, density, degree distribution, clustering, transitivity, assortativity, connectivity, and efficiency. A class-balanced random forest is evaluated with stratified cross-validation. This workflow requires independent graphs—not nodes from one graph—as samples. The included demo contrasts synthetic small-world and preferential-attachment graph families.

## Differential rich club

Normalized curves are estimated independently for both networks with separate seeded degree-preserving null ensembles. NodeSafari reports the difference at shared degree thresholds. This descriptive comparison is not a replicate-aware inferential test.

## Reproducibility

Layouts, random null networks, perturbation repeats, cross-validation splits, and models use fixed random seeds. Exported tables contain the values displayed by the application.
