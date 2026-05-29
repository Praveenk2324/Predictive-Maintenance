from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware          # FIX 4: CORS import
from pydantic import BaseModel
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from mlflow.tracking import MlflowClient
import torch
import joblib
import mlflow
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

from src.api.database import get_db, engine
from src.api import models

# --- Global Model Variables ---
ml_models = {}
MLFLOW_URI = "http://mlflow:5000"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANOMALY_THRESHOLD = 0.253008

# FIX 2: REMOVED models.Base.metadata.create_all(bind=engine) from module level.
# Calling it at import time crashes uvicorn if postgres isn't ready yet.
# It is now safely inside the lifespan function below.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs exactly once when the server starts.
    Loads all ML models into memory using MLflow aliases.
    """
    print(f"Starting API... Assigning tensors to: {DEVICE}")

    # FIX 2: Create DB tables here, inside lifespan, AFTER postgres is confirmed
    # healthy by docker-compose health checks. Safe to call multiple times (idempotent).
    print("Ensuring database tables exist...")
    models.Base.metadata.create_all(bind=engine)

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    print("Loading Production LSTM Autoencoder from Registry...")
    lstm_name = "NASA_LSTM_Autoencoder"
    alias = "production"

    prod_lstm_version = client.get_model_version_by_alias(lstm_name, alias)
    scaler_path = client.download_artifacts(prod_lstm_version.run_id, 'sensor_scaler.pkl')
    ml_models['scaler'] = joblib.load(scaler_path)

    lstm_model = mlflow.pytorch.load_model(f"models:/{lstm_name}@{alias}", map_location='cpu')
    lstm_model.to(DEVICE)
    lstm_model.eval()
    ml_models['lstm'] = lstm_model

    print("Loading Production Random Forest Regressor from Registry...")
    rf_name = "NASA_RandomForest_RUL"
    ml_models['rf'] = mlflow.sklearn.load_model(f"models:/{rf_name}@{alias}")

    print("Connecting to ChromaDB Vector Space...")
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    ml_models['rag'] = chroma_client.get_collection(
        name="maintenance_knowledge_base",
        embedding_function=sentence_transformer_ef
    )

    print("--- API SUCCESSFULLY INITIALIZED ---")
    yield

    ml_models.clear()


app = FastAPI(title="Aerospace Predictive Maintenance API", version="1.0", lifespan=lifespan)

# FIX 4: CORS middleware — must be added right after app = FastAPI(...)
# Allows the HTML frontend (served from any origin) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Accepts requests from any origin (browser, Streamlit, HTML dashboard)
    allow_credentials=True,
    allow_methods=["*"],        # POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)


class EnginePayload(BaseModel):
    engine_id: int
    rf_features: list[float]
    lstm_sequence: list[list[float]]


@app.post("/predict")
def predict_engine_status(payload: EnginePayload, db: Session = Depends(get_db)):
    """
    Receives engine telemetry, calculates RUL, detects anomalies,
    queries RAG if failing, and persists the log to PostgreSQL.
    """
    try:
        rf_input = np.array(payload.rf_features).reshape(1, -1)
        predicted_rul = float(ml_models['rf'].predict(rf_input)[0])

        seq_array = np.array(payload.lstm_sequence)
        scaled_seq = ml_models['scaler'].transform(seq_array)
        tensor_seq = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            reconstruction = ml_models['lstm'](tensor_seq)
            mse = torch.mean((reconstruction - tensor_seq) ** 2).item()

        is_anomaly = "ANOMALY" if mse > ANOMALY_THRESHOLD else "NORMAL"

        if is_anomaly == "ANOMALY":
            query_text = (
                f"Engine {payload.engine_id} showing severe sensor deviation. "
                "High reconstruction error indicating critical failure."
            )
            results = ml_models['rag'].query(query_texts=[query_text], n_results=1)
            if results['documents'] and results['documents'][0]:
                rag_context = results['documents'][0][0]
            else:
                rag_context = "Anomaly detected, but no matching historical logs found."
        else:
            rag_context = "System operating within normal parameters. Continue standard monitoring."

        new_log = models.PredictionLog(
            engine_id=payload.engine_id,
            predicted_rul=predicted_rul,
            reconstruction_mse=mse,
            is_anomaly=is_anomaly,
            rag_diagnostics=rag_context
        )
        db.add(new_log)
        db.commit()
        db.refresh(new_log)

        return {
            "prediction_id": new_log.id,
            "engine_id": payload.engine_id,
            "status": "🚨 CRITICAL FAULT" if is_anomaly == "ANOMALY" else "✅ HEALTHY",
            "predicted_rul_cycles": round(predicted_rul, 2),
            "anomaly_mse": round(mse, 4),
            "anomaly_threshold": round(ANOMALY_THRESHOLD, 4),   # FIX 5: was missing, showed N/A in UI
            "maintenance_recommendation": rag_context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{engine_id}")
def get_engine_history(engine_id: int, db: Session = Depends(get_db)):
    """
    Retrieves the last 10 prediction logs for a specific engine from PostgreSQL.
    """
    records = (
        db.query(models.PredictionLog)
        .filter(models.PredictionLog.engine_id == engine_id)
        .order_by(models.PredictionLog.timestamp.desc())
        .limit(10)
        .all()
    )
    if not records:
        raise HTTPException(status_code=404, detail="No logs found for this engine.")

    return [
        {
            "prediction_id": r.id,
            "timestamp": r.timestamp,
            "status": "🚨 CRITICAL FAULT" if r.is_anomaly == "ANOMALY" else "✅ HEALTHY",
            "predicted_rul_cycles": round(r.predicted_rul, 2),
            "anomaly_mse": round(r.reconstruction_mse, 4),
            "anomaly_threshold": round(ANOMALY_THRESHOLD, 4),
            "maintenance_recommendation": r.rag_diagnostics
        }
        for r in records
    ]