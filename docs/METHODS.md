# Methods and interpretation

NodeSafari is designed for exploratory and hypothesis-generating analysis. It does not infer molecular causality.

## Rich-club analysis

For degree threshold `k`, the rich-club coefficient is the density of the subgraph induced by nodes with degree greater than `k`. The normalized coefficient divides the observed density by the mean density from degree-preserving randomized networks. Values greater than one suggest enrichment relative to the selected null model.

## Communities

Communities are estimated with greedy modularity maximization. Community labels are arbitrary identifiers and should not be interpreted as biological classes without external validation.

## Perturbation screen

Each node is removed once. The structural impact score combines loss of global efficiency (60%) and additional loss of the largest connected component (40%). This is a transparent prioritization heuristic, not a simulation of knockout biology.

## Spectral embeddings

Node vectors are learned by truncated singular-value decomposition of the symmetrically normalized weighted adjacency matrix. Nearby nodes have similar structural contexts; similarity does not necessarily imply shared molecular function.

## Link prediction

Candidate missing edges are ranked using an ensemble of common-neighbor count, Jaccard similarity, and log-transformed Adamic–Adar score. These candidates require independent experimental evidence.

## Reproducibility

Layouts, random null networks, and embeddings use fixed random seeds. Exported tables contain the values displayed by the application.
