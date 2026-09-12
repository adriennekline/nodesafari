# Getting started with NodeSafari

This guide assumes no Python or Docker experience. You need only one of the two installation methods below.

## Choose how to run NodeSafari

| Method | Best for | What it installs |
|---|---|---|
| Docker | Most users, shared workstations, and servers | Everything inside an isolated container |
| Local Python | Developers who want to change the source code | Packages inside a local `.venv` folder |

Docker does **not** create a `.venv`. It provides its own isolated environment inside the container, so a virtual environment is unnecessary when using Docker.

## Option 1: Docker

### 1. Install Docker

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) on macOS or Windows. On Linux, install Docker Engine and the Docker Compose plugin using your distribution's instructions.

Open Docker Desktop and wait until it reports that Docker is running.

### 2. Download NodeSafari

Open Terminal on macOS/Linux or PowerShell on Windows:

```bash
git clone https://github.com/adriennekline/nodesafari.git
cd nodesafari
```

If `git` is unavailable, use **Code → Download ZIP** on the repository page, extract the archive, and open a terminal inside that folder.

### 3. Build and start the app

```bash
docker compose up --build -d
```

The first build may take several minutes while Docker downloads Python and installs the required packages.

### 4. Open NodeSafari

Visit [http://localhost:8501](http://localhost:8501) in a web browser.

### 5. Stop the app

```bash
docker compose down
```

Your uploaded network files are not stored in the container after the session.

## Option 2: Local Python

Install Python 3.10 or newer, then open a terminal in the NodeSafari folder.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The `.venv` folder is created by the first command. It is intentionally excluded from Git because it is machine-specific and can be recreated from `requirements.txt`.

## Try the example networks

NodeSafari opens with synthetic control and disease-like interaction networks. No upload is required to explore the five workspaces:

1. **Explore** — network view, modules, and high-interest nodes
2. **Rich club** — normalized rich-club organization
3. **Compare** — changes between two networks
4. **Perturb** — structural effects of single-node deletion
5. **ML lab** — node embeddings, similar nodes, and candidate missing interactions

## Upload your own data

Create a CSV file with at least `source` and `target` columns. An optional numeric `weight` column represents interaction strength.

```csv
source,target,weight
TP53,MDM2,2.8
TP53,ATM,2.2
ATM,CHEK2,2.1
```

Use **Primary network** in the sidebar for the network you want to analyze. Use **Comparison network** only when comparing two conditions.

## Common problems

### The browser says the site cannot be reached

Confirm that Docker Desktop is running and inspect the service:

```bash
docker compose ps
docker compose logs nodesafari
```

### Port 8501 is already in use

Change the first port in `compose.yaml` from `8501:8501` to `8502:8501`, restart the container, and open `http://localhost:8502`.

### A CSV is rejected

Check that the column names are exactly `source`, `target`, and optionally `weight`. Weight values must be numeric. Remove protected or identifiable data before using a public deployment.

### Start over with a clean Docker build

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Updating NodeSafari

From the repository folder:

```bash
git pull
docker compose up --build -d
```
