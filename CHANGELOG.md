# Changelog

## 1.1.0 — 2026-09-13

- Added an end-to-end two-layer GCN for cross-validated node classification.
- Added a GCN graph autoencoder for held-out-edge evaluation and missing-link ranking.
- Added a pooled two-layer GCN for cross-validated whole-graph classification.
- Added model selectors and neural training curves while retaining random forests as interpretable baselines.
- Kept the neural stack CPU-only and dependency-light with a NumPy implementation.
- Expanded automated coverage to include all three neural workflows.

## 1.0.0 — 2026-09-13

- Added a complete upload-to-QC workflow with downloadable input reports.
- Added k-core/core–periphery roles, articulation nodes, bridge edges, and expanded network statistics.
- Added differential community alignment, partition-similarity scores, and differential normalized rich-club curves.
- Added single-edge deletion and repeated targeted-versus-random robustness analysis.
- Added cross-validated node prediction, held-out-edge link prediction, and graph classification with validation metrics and feature importance.
- Reorganized the application into Data & QC, Explore, Compare, Perturb, and ML workspaces.
- Added synthetic node labels, graph-family demos, downloads throughout the application, and v1.0 method documentation.
- Expanded automated coverage to 12 tests across input, structure, comparison, perturbation, and ML workflows.

## 0.1.0 — 2026-09-12

- Added validated weighted edge-list import.
- Added network summaries, hub metrics, modularity communities, and visualization.
- Added normalized rich-club analysis with degree-preserving null networks.
- Added differential network comparison and node rewiring tables.
- Added in-silico single-node perturbation screening.
- Added spectral node embeddings, similar-node discovery, and link prediction.
- Added CSV exports, synthetic examples, tests, citation metadata, and contributor guidance.
- Added a non-root Docker image, health check, hardened Compose configuration, and self-hosting guidance.
- Added beginner-friendly Docker and local Python instructions plus BibTeX citation metadata.
- Added a prominent three-step Docker start path aligned with RichClub Explorer.
