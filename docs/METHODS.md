# Methods and interpretation

NodeSafari is designed for exploratory and hypothesis-generating analysis. It does not infer causality.

## Rich-club analysis

For threshold $k$, let $N_{>k}$ be the number of nodes whose selected richness is
strictly greater than $k$, and let $E_{>k}$ be the number of edges among those
nodes. The binary coefficient is

$$
\phi(k)=\frac{2E_{>k}}{N_{>k}(N_{>k}-1)}.
$$

This is the density of the subgraph induced by the richer nodes. Richness may be
defined as degree. In weighted mode it may instead be defined as strength, the sum
of incident edge weights.

The weighted coefficient is the sum of weights among rich nodes divided by the sum
of the globally strongest $E_{>k}$ edge weights. This is an Opsahl-style weighted
formulation. Weighted nulls preserve the degree sequence and global weight
distribution by randomly permuting weights over rewired edges; they do not preserve
each node's strength, so weighted inference is explicitly exploratory.

Each null network is generated through double-edge swaps. The number of attempted
swaps is the selected swaps-per-edge value multiplied by the observed edge count.
This preserves every node's degree while randomizing which node pairs are connected.
The normalized coefficient is

$$
\rho(k)=\frac{\phi_{\mathrm{obs}}(k)}
{B^{-1}\sum_{b=1}^{B}\phi^{(b)}_{\mathrm{null}}(k)}.
$$

The 95% null envelope is the 2.5th to 97.5th percentile of the null coefficients.
The one-sided empirical p-value uses a plus-one correction:

$$
p(k)=\frac{1+\sum_{b=1}^{B}
\mathbf{1}[\phi^{(b)}_{\mathrm{null}}(k)\geq\phi_{\mathrm{obs}}(k)]}{B+1}.
$$

NodeSafari also reports Benjamini-Hochberg q-values as descriptive multiplicity
information. Thresholds are nested and their tests are therefore dependent; q-values
do not replace sensitivity analysis across a meaningful, contiguous threshold range.
A threshold is marked as an exploratory signal only when $\rho>1$, empirical
$p<0.05$, and the selected minimum number of rich nodes remains. At least 1,000 null
networks are recommended for final inference; smaller ensembles are for rapid
exploration.

At an inspected threshold, edges are classified as rich-club when both endpoints
are rich, feeder when exactly one endpoint is rich, and local when neither endpoint
is rich. These roles describe topology and do not imply causal or domain function.

## Communities

Communities are estimated with greedy modularity maximization. Community labels are arbitrary identifiers and should not be interpreted as domain classes without external validation.

For two-network comparison, NodeSafari calculates adjusted Rand index and normalized mutual information on shared nodes. Community labels in network B are greedily aligned to network A by node overlap before the fraction of reassigned nodes is reported. This alignment aids interpretation; it is not a statistical test across replicated observations.

## Core–periphery and bridges

Core position is calculated with k-core decomposition after removing self-loops. The normalized core score divides each node's core number by the maximum observed core number. This is a discrete structural decomposition, not a fitted continuous core–periphery block model.

Articulation nodes and bridge edges are exact disconnection points in the undirected projection. Removal consequences are reported using connected-component counts and sizes.

## Perturbation screen

Each node or edge is removed once. The structural impact score combines loss of global efficiency (60%) and loss of the largest connected component (40%). This is a transparent prioritization heuristic, not a simulation of knockout biology.

Robustness analysis compares adaptive removal of the current highest-degree node with repeated random node removal. The random curve reports its mean and standard deviation. Largest-component size is normalized to the original node count.

## Spectral embeddings

Node vectors are learned by truncated singular-value decomposition of the symmetrically normalized weighted adjacency matrix. Nearby nodes have similar structural contexts; similarity does not necessarily imply shared function.

## Node prediction

Node prediction combines centrality, clustering, spectral embeddings, and any uploaded numeric covariates. A class-balanced random forest is evaluated with stratified k-fold cross-validation, limited by the smallest class. The displayed predictions are out-of-fold. Accuracy, balanced accuracy, and macro F1 are reported.

## Link prediction

Observed non-bridge edges are held out, and a class-balanced random forest learns from common neighbors, Jaccard similarity, Adamic–Adar, preferential attachment, and endpoint degrees in the remaining graph. Recovery is evaluated with ROC AUC and average precision. The fitted model then ranks genuinely absent interactions. Small networks fall back to a structural ensemble. All candidates require independent experimental evidence.

## Graph classification

Each uploaded graph becomes one sample described by size, density, degree distribution, clustering, transitivity, assortativity, connectivity, and efficiency. A class-balanced random forest is evaluated with stratified cross-validation. This workflow requires independent graphs—not nodes from one graph—as samples. The included demo contrasts synthetic small-world and preferential-attachment graph families.

## Graph neural networks

NodeSafari also provides three CPU-oriented neural alternatives implemented directly with NumPy:

- **Node classification:** a two-layer graph convolutional network (GCN) propagates structural and optional numeric node features through the normalized adjacency matrix. Evaluation is transductive and stratified: the full topology and node features are visible, but labels in each test fold are withheld during training.
- **Link prediction:** a two-layer GCN encoder learns node embeddings by reconstructing observed edges against sampled absent pairs. A dot-product decoder scores held-out edges and candidate missing interactions. Evaluation reports ROC AUC and average precision on balanced held-out positive and negative pairs.
- **Graph classification:** a shared two-layer GCN generates node embeddings for each independent graph, mean pooling creates a graph representation, and a neural classification head predicts the graph label. Evaluation uses stratified graph-level folds.

All three models use ReLU activations, cross-entropy or binary cross-entropy objectives, L2 regularization, Adam optimization, fixed random seeds, and class weighting where applicable. Training-loss curves are diagnostic only; model selection should rely on held-out performance and external domain validation.

## Differential rich club

Normalized curves are estimated independently for both networks with separate seeded degree-preserving null ensembles. NodeSafari reports the difference at shared degree thresholds. This descriptive comparison is not a replicate-aware inferential test.

## Reproducibility

Layouts, random null networks, perturbation repeats, cross-validation splits, and models use fixed random seeds. Exported tables contain the values displayed by the application.
