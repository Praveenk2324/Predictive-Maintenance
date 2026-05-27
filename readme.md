# ✈️ Aerospace Predictive Maintenance System
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![AWS](https://img.shields.io/badge/AWS-EC2%20Deployed-FF9900.svg)

An end-to-end MLOps pipeline for jet engine health monitoring, combining a Random Forest RUL (Remaining Useful Life) regressor, an LSTM Autoencoder for anomaly detection, and a RAG (Retrieval-Augmented Generation) system powered by historical maintenance logs — all served via a FastAPI backend, tracked with MLflow, and visualized through a Streamlit dashboard.

---
### 🌐 **Live Link:** [Predictive Maintenance System](http://<YOUR_AWS_PUBLIC_IP>:8000/docs)
*(Note: Hosted on a dedicated AWS EC2 instance)

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [Deployment on AWS EC2](#deployment-on-aws-ec2)
- [Local Development](#local-development)
- [DVC Pipeline](#dvc-pipeline)
- [Environment Variables](#environment-variables)
- [Dataset](#dataset)

---

## Architecture Overview

```
Raw Sensor Data (NASA CMAPSS)
        │
        ▼
┌─────────────────────┐
│  Feature Engineering │  ← DVC Stage 1 (src/features.py)
│  Rolling stats,      │    Rolling mean/std/diff across
│  windowed sensors    │    windows of 5, 10, 30 cycles
└────────┬────────────┘
         │
    ┌────┴──────┐
    ▼           ▼
┌──────────┐  ┌─────────────────────┐
│  Random  │  │  LSTM Autoencoder   │
│  Forest  │  │  (Anomaly Detection)│
│  (RUL    │  │  Trained on healthy │
│Regressor)│  │  sequences only     │
└────┬─────┘  └──────────┬──────────┘
     │                   │
     └─────────┬─────────┘
               │
               ▼
        ┌─────────────┐      ┌──────────────────────┐
        │  FastAPI     │◄────►│  ChromaDB (RAG)      │
        │  /predict    │      │  1,500 Maintenance   │
        │  /history    │      │  Logs as Embeddings  │
        └──────┬───────┘      └──────────────────────┘
               │
        ┌──────┴───────┐
        │  PostgreSQL  │  ← Persists all prediction logs
        └──────┬───────┘
               │
        ┌──────┴───────┐
        │  Streamlit   │  ← Fleet dashboard + time-travel slider
        └──────────────┘
```

All services are containerized with Docker Compose and deployed on an AWS EC2 instance. ML experiments and model versioning are managed with MLflow.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Pipeline | DVC, Pandas, NumPy |
| ML — RUL Regression | Scikit-learn (Random Forest), Optuna (HPO) |
| ML — Anomaly Detection | PyTorch (LSTM Autoencoder) |
| ML — RAG | ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`) |
| Experiment Tracking | MLflow (SQLite backend, model registry with aliases) |
| API | FastAPI, Uvicorn, Pydantic |
| Database | PostgreSQL 15 (via SQLAlchemy) |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Deployment | AWS EC2 (Ubuntu) |

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   ├── turbofan/          # NASA CMAPSS dataset files
│   │   │   ├── train_FD001.txt
│   │   │   ├── test_FD001.txt
│   │   │   └── RUL_FD001.txt
│   │   └── maintenance_logs.csv   # Generated synthetic logs
│   └── processed/
│       ├── features_train.parquet
│       └── features_test.parquet
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI app, lifespan model loading
│   │   ├── database.py        # SQLAlchemy engine + session
│   │   └── models.py          # PredictionLog ORM model
│   ├── data/
│   │   └── generate_logs.py   # Synthetic maintenance log generator
│   ├── features.py            # Feature engineering pipeline
│   └── models/
│       ├── rul_regressor.py   # Random Forest + Optuna tuning
│       ├── lstm_autoencoder.py # LSTM Autoencoder training
│       ├── build_vector_db.py  # ChromaDB ingestion
│       └── rag_query.py       # RAG query utility
├── artifacts/                 # Saved plots and scaler
├── chroma_db/                 # Persisted ChromaDB vector store
├── mlruns/                    # MLflow run artifacts
├── streamlit_app.py           # Fleet monitoring dashboard
├── init_db.py                 # Creates PostgreSQL tables
├── dvc.yaml                   # DVC pipeline definition
├── docker-compose.yml         # Multi-service orchestration
├── Dockerfile                 # API container
└── requirements.txt
```

---

## ML Pipeline

### 1. Feature Engineering (`src/features.py`)

Processes the NASA CMAPSS FD001 dataset. Drops zero-variance sensors and engineers rolling features for each of the 14 useful sensors across windows of 5, 10, and 30 cycles:

- **Rolling Mean** — smooths sensor trends
- **Rolling Std** — captures variance
- **Rolling Diff** — captures rate of change

RUL is computed for every cycle and clipped at 125 (piecewise-linear target). Final shape: ~20,000 rows × ~430 features.

### 2. RUL Regressor (`src/models/rul_regressor.py`)

- **Model**: Random Forest Regressor
- **Tuning**: Optuna TPE sampler, 50 trials, minimizing RMSE
- **Search space**: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- **Logged to MLflow**: params, RMSE, R², feature importance plot, model artifact
- **Registry alias**: `production` under `NASA_RandomForest_RUL`

### 3. LSTM Autoencoder (`src/models/lstm_autoencoder.py`)

- **Architecture**: Encoder LSTM → compressed hidden state → Decoder LSTM → Linear output
- **Training data**: Only healthy sequences (RUL > 80), sequence length = 30 cycles
- **Loss**: MSE reconstruction error
- **Anomaly threshold**: 95th percentile of reconstruction error on the healthy training set (~0.253)
- **Artifacts saved**: `sensor_scaler.pkl`, reconstruction error distribution plot
- **Registry alias**: `production` under `NASA_LSTM_Autoencoder`

### 4. RAG System (`src/models/build_vector_db.py`)

- 1,500 synthetic maintenance logs generated across 18 aerospace components (ATA chapters 36–79)
- Embedded using `sentence-transformers/all-MiniLM-L6-v2`
- Stored in ChromaDB as a persistent collection (`maintenance_knowledge_base`)
- At inference: anomalous engines trigger a semantic search for the most relevant historical maintenance log, returned as the `maintenance_recommendation` field

---

## API Reference

Base URL: `http://<EC2_PUBLIC_IP>:8000`

### `POST /predict`

Receives engine telemetry, computes RUL and anomaly score, queries RAG if anomalous, and persists the result to PostgreSQL.

**Request Body:**
```json
{
  "engine_id": 42,
  "rf_features": [/* 140 floats: last 10 cycles × 14 sensors, flattened */],
  "lstm_sequence": [/* 30 × 14 nested float array */]
}
```

**Response:**
```json
{
  "prediction_id": 17,
  "engine_id": 42,
  "status": "🚨 CRITICAL FAULT",
  "predicted_rul_cycles": 14.83,
  "anomaly_mse": 0.4821,
  "anomaly_threshold": 0.2530,
  "maintenance_recommendation": "ATA 72-50 | Engine 42: HPT stage-1 blade TBC spallation..."
}
```

**Status values:**
- `✅ HEALTHY` — MSE ≤ 0.253
- `🚨 CRITICAL FAULT` — MSE > 0.253

---

### `GET /history/{engine_id}`

Returns the 10 most recent prediction logs for a given engine from PostgreSQL.

**Response:** Array of prediction objects (same schema as `/predict` response), ordered by timestamp descending.

---

## Deployment on AWS EC2

### Prerequisites

- EC2 instance running **Ubuntu 22.04 LTS** (recommended: `t3.medium` or larger)
- Security group inbound rules:
  - Port `8000` — FastAPI
  - Port `5000` — MLflow UI
  - Port `8501` — Streamlit
  - Port `5433` — PostgreSQL (optional, for direct DB access)
- Docker and Docker Compose installed

### Step 1 — SSH into your instance

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### Step 2 — Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

### Step 3 — Clone the repository and prepare data

```bash
git clone https://github.com/your-org/aerospace-pm.git
cd aerospace-pm

# Place the NASA CMAPSS dataset files in data/raw/turbofan/
# train_FD001.txt, test_FD001.txt, RUL_FD001.txt
```

### Step 4 — Run the DVC pipeline

```bash
pip install dvc
dvc repro
```

This runs feature engineering, trains both models, and logs everything to MLflow.

### Step 5 — Train models and register them in MLflow

After `dvc repro` completes, open the MLflow UI (`http://localhost:5000`) and set the `production` alias on the latest run for both:
- `NASA_RandomForest_RUL`
- `NASA_LSTM_Autoencoder`

Alternatively, do it via the Python client:
```python
from mlflow.tracking import MlflowClient
client = MlflowClient("http://localhost:5000")
client.set_registered_model_alias("NASA_RandomForest_RUL", "production", "1")
client.set_registered_model_alias("NASA_LSTM_Autoencoder", "production", "1")
```

### Step 6 — Generate maintenance logs and build the vector DB

```bash
python src/data/generate_logs.py
python src/models/build_vector_db.py
```

### Step 7 — Initialise the PostgreSQL schema

```bash
python init_db.py
```

### Step 8 — Launch all services

```bash
docker compose up -d --build
```

This starts:
- `pm_api` — FastAPI on port 8000
- `pm_postgres` — PostgreSQL on port 5433
- `pm_mlflow` — MLflow tracking server on port 5000

### Step 9 — Launch the Streamlit dashboard

```bash
pip install streamlit
streamlit run streamlit_app.py --server.port 8501
```

Access the dashboard at `http://<EC2_PUBLIC_IP>:8501`

### Verify everything is running

```bash
docker ps
curl http://localhost:8000/docs   # FastAPI Swagger UI
```

---

## Local Development

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full DVC pipeline
dvc repro

# Start infrastructure only
docker compose up postgres mlflow -d

# Initialize the database
python init_db.py

# Start the API (outside Docker, pointing at localhost services)
# Update DB_HOST in database.py to 'localhost' for local dev
uvicorn src.api.main:app --reload --port 8000

# Launch Streamlit
streamlit run streamlit_app.py
```

---

## DVC Pipeline

The `dvc.yaml` defines three stages with full dependency tracking:

```
feature_engineering
    → train_df_model (Random Forest)
    → train_lstm_model (LSTM Autoencoder)
```

Reproduce any stage:
```bash
dvc repro                        # full pipeline
dvc repro train_lstm_model       # single stage
dvc dag                          # visualize DAG
```

Stage outputs are tracked in `.gitignore` via `artifacts/.gitignore` — plots and the scaler are committed only when explicitly added.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `postgres` | PostgreSQL hostname (use `localhost` for local dev) |
| `MLFLOW_URI` | `http://mlflow:5000` | MLflow tracking server URI |
| `ANOMALY_THRESHOLD` | `0.253008` | LSTM MSE threshold (hardcoded in `main.py`, set from training) |

---

## Dataset

This project uses the **NASA CMAPSS Turbofan Engine Degradation Simulation** dataset (FD001 subset).

- **Train set**: 100 engines run to failure, ~20,000 cycles total
- **Test set**: 100 engines with ground-truth RUL labels
- **Sensors used**: 14 of 21 (zero-variance sensors dropped)
- **Target**: RUL clipped at 125 cycles (piecewise-linear assumption)

Download from: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Place files at:
```
data/raw/turbofan/train_FD001.txt
data/raw/turbofan/test_FD001.txt
data/raw/turbofan/RUL_FD001.txt
```

---

## MLflow Model Registry

Both models are registered and versioned in MLflow. The API loads models exclusively via **aliases** (not version numbers), so promoting a new model to production requires only:

```python
client.set_registered_model_alias("NASA_RandomForest_RUL", "production", "<new_version>")
```

No API restart required if models are reloaded on startup via the `lifespan` handler.

---

## Streamlit Dashboard Features

- **Engine selector** — choose any of the 100 training engines
- **Time-travel slider** — simulate the engine at any historical cycle (min 30 due to LSTM sequence requirement)
- **Live diagnostics** — calls `/predict` and displays status, RUL, anomaly MSE vs threshold
- **RAG recommendation** — surfaces the most semantically relevant historical maintenance log when anomaly is detected
- **PostgreSQL history** — fetches and displays the last 10 logged predictions for any engine