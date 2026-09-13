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

#### Windows

1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/).
2. Accept the WSL 2 installation or update prompts if Docker displays them.
3. Restart the computer if requested.
4. Open Docker Desktop and wait until it reports that Docker is running.

#### macOS

1. Install [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/).
2. Choose the Apple silicon or Intel installer appropriate for the Mac.
3. Open Docker Desktop and wait until it reports that Docker is running.

#### Linux

Install [Docker Engine](https://docs.docker.com/engine/install/) and the Docker
Compose plugin for your Linux distribution. Start the Docker service before
continuing. If Docker reports a permission error, follow Docker's
[Linux post-installation instructions](https://docs.docker.com/engine/install/linux-postinstall/)
or contact your system administrator.

### 2. Download NodeSafari

Open Terminal on macOS/Linux or PowerShell on Windows:

```bash
git clone https://github.com/adriennekline/nodesafari.git
cd nodesafari
```

If `git` is unavailable, use **Code → Download ZIP** on the repository page and
extract the archive. Then open a terminal inside that folder:

- **Windows:** Open the folder in File Explorer, click the address bar, type
  `powershell`, and press Enter.
- **macOS:** Open Terminal, type `cd ` with a trailing space, drag the extracted
  folder into Terminal, and press Enter.
- **Linux:** Open the folder in your file manager, right-click inside it, and
  select **Open in Terminal** when available.

### 3. Build and start the app

```bash
docker compose up --build
```

The first build may take several minutes while Docker downloads Python and installs the required packages.
Keep this terminal open while using NodeSafari.

### 4. Open NodeSafari

Visit [http://localhost:8501](http://localhost:8501) in a web browser.

### 5. Stop the app

Press `Ctrl+C` in the terminal running NodeSafari, then run:

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

1. **Data & QC** — input validation, cleaning warnings, and connectivity checks
2. **Explore** — hubs, rich club, communities, k-core position, bridges, and statistics
3. **Compare** — whole-network, hub, community, and rich-club changes
4. **Perturb** — node deletion, edge deletion, and robustness curves
5. **ML** — embeddings plus random-forest and graph-neural-network prediction

Start in **Data & QC** to confirm that the example inputs pass validation, then use
**Explore** to inspect the network and ranked nodes. Open **Compare** to see how the
included disease-like network differs from the control network. The **ML** workspace
includes synthetic labels and a generated graph dataset, so its workflows run before
you upload supervised-learning data. Use the **Model** controls to switch between
interpretable random-forest baselines and the neural models. Analysis tables can be
downloaded as CSV.

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

### `docker: command not found`

Docker is not installed or the terminal has not recognized the installation.
Install Docker, restart the terminal, and try again.

### `Cannot connect to the Docker daemon`

Docker Desktop or Docker Engine is not running. Start it, wait for the engine to
become ready, and rerun the command.

### `docker compose` is not recognized

Update Docker Desktop or install the Docker Compose plugin. The current command is
`docker compose` with a space. Older systems may use `docker-compose` with a hyphen.

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

### Apple silicon computer

No configuration change should be necessary. The official Python image supports
both Apple silicon and standard x86-64 computers.

### Institutional network or proxy error

The first build downloads a base image and Python packages. An institutional
firewall or proxy may block those downloads. Ask local IT for the institution's
approved Docker proxy configuration.

### Start over with a clean Docker build

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Restart and run in the background

Restart without rebuilding when the files have not changed:

```bash
docker compose up
```

Run NodeSafari without keeping the terminal attached:

```bash
docker compose up --detach
```

The application remains available at [http://localhost:8501](http://localhost:8501).
Stop a detached application with `docker compose down`.

## Updating NodeSafari

From the repository folder:

```bash
git pull --ff-only
docker compose up --build
```

If NodeSafari was downloaded as a ZIP, download the newest ZIP into a new folder
and run `docker compose up --build` from that folder.

## Data handling and privacy

The provided Compose configuration runs NodeSafari locally and does not create a
persistent Docker volume for uploaded data. Uploaded tables are processed inside
the running application and are not intentionally written to persistent container
storage. The original file remains wherever it was saved on the computer.

Do not expose port 8501 publicly or use confidential or regulated data in an
externally hosted deployment without appropriate institutional review, access
controls, and safeguards.

## Remove NodeSafari from Docker

Stop and remove the container:

```bash
docker compose down
```

Optionally remove the locally built image:

```bash
docker image rm nodesafari:latest
```

The downloaded project folder can then be deleted normally. Removing NodeSafari
does not uninstall Docker Desktop.
