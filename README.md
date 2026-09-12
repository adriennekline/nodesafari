# NodeSafari

**Explore biological networks. Return with testable hypotheses.**

NodeSafari is an open-source Streamlit application and Python package for researchers who want to explore biological networks without assembling a custom graph-analysis pipeline. Rich-club organization is included, but the expedition goes further: discover structure, compare conditions, simulate perturbations, and use graph-based machine learning to prioritize follow-up questions.

> **Status:** research alpha. Outputs are exploratory and do not establish biological causality.

![NodeSafari interface showing a biological interaction network, analysis metrics, and ranked nodes](docs/interface-preview.svg)

## What researchers can do

| Question | NodeSafari workflow |
|---|---|
| Which nodes and modules organize the network? | Centrality, communities, connectivity, and network statistics |
| Are highly connected nodes unusually interconnected? | Normalized rich-club analysis with degree-preserving null networks |
| What changes between control and disease? | Whole-network and node-level differential comparison |
| Which node deletion most disrupts organization? | Transparent perturbation screen based on efficiency and fragmentation |
| Which nodes occupy similar network roles? | Spectral node embeddings and nearest-node search |
| Which interactions may be missing? | Interpretable ensemble link prediction |

## Quick start

### Docker (recommended for labs and servers)

```bash
git clone https://github.com/adriennekline/nodesafari.git
cd nodesafari
docker compose up --build -d
```

Open `http://localhost:8501`. The container runs as a non-root user, exposes a health check, uses a read-only filesystem, and retains uploaded data only for the current application session.

To stop it:

```bash
docker compose down
```

### Local Python

```bash
git clone https://github.com/adriennekline/nodesafari.git
cd nodesafari
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The application opens with two synthetic example networks, so every workflow can be explored before uploading data.

## Input format

Upload a CSV edge list with `source` and `target` columns. `weight` is optional.

```csv
source,target,weight
TP53,MDM2,2.8
TP53,ATM,2.2
ATM,CHEK2,2.1
```

- Node identifiers may be gene, protein, metabolite, cell, brain-region, or other labels.
- Duplicate edges are combined by summing their weights.
- Self-loops and rows with missing endpoints are removed.
- Do not upload identifiable or otherwise restricted data to a public deployment.

## Use the analysis package directly

```python
import pandas as pd
from nodesafari import graph_from_edgelist, perturbation_screen

edges = pd.read_csv("my_edges.csv")
graph = graph_from_edgelist(edges)
ranked_nodes = perturbation_screen(graph)
print(ranked_nodes.head())
```

## Scientific guardrails

NodeSafari deliberately labels its outputs as structural candidates or predictions. A high-impact node is not automatically a therapeutic target, and a predicted link is not evidence of a molecular interaction. See [Methods and interpretation](docs/METHODS.md) for assumptions and limitations.

## Self-hosting

The included container can run on a workstation, institutional server, or any Docker-compatible hosting platform. For remote deployment, place it behind your institution's authenticated reverse proxy and TLS termination. The app does not persist uploaded files, but deployment controls must still match the governance requirements for the data being analyzed.

## Roadmap

- Statistical testing for differential networks across biological replicates
- Temporal-network analysis
- Node metadata and pathway enrichment
- Supervised node, link, and graph prediction
- Heterogeneous and multilayer biological graphs
- Counterfactual analysis with uncertainty estimates

## Contributing

Method requests and contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), which asks contributors to frame features around a scientific question and document assumptions.

## Attribution

Created by **Adrienne Kline** with **Northwestern University**. Licensed under the [MIT License](LICENSE). Citation metadata are provided in [CITATION.cff](CITATION.cff).
