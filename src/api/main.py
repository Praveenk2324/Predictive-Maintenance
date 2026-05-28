from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
# mlflow.set_tracking_uri("http://mlflow:5000")
MLFLOW_URI = "http://mlflow:5000"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANOMALY_THRESHOLD = 0.253008

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs exactly once when the server starts.
    It loads all machine learning models into memory using modern MLflow Aliases.
    """
    print(f"Starting API... Assigning tensors to: {DEVICE}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    print("Loading Production LSTM Autoencoder from Registry...")
    lstm_name = "NASA_LSTM_Autoencoder"
    alias = "production"

    prod_lstm_version = client.get_model_version_by_alias(lstm_name, alias)
    # scaler_path = client.download_artifacts(prod_lstm_version.run_id, 'sensor_scaler.pkl')
    # ml_models['scaler'] = joblib.load(scaler_path)
    ml_models['scaler'] = joblib.load('sensor_scaler.pkl')

    lstm_model = mlflow.pytorch.load_model(f"models:/{lstm_name}@{alias}")
    lstm_model.to(DEVICE)
    lstm_model.eval()
    ml_models['lstm'] = lstm_model

    print("Loading Production Random Forest Regressor from Registry...")
    rf_name = "NASA_RandomForest_RUL"
    ml_models['rf'] = mlflow.sklearn.load_model(f"models:/{rf_name}@{alias}")

    print("Connecting to ChromaDB Vector Space...")
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    ml_models['rag'] = chroma_client.get_collection(name="maintenance_knowledge_base", embedding_function=sentence_transformer_ef)

    print("--- API SUCCESSFULLY INITIALIZED ---")
    yield
    
    ml_models.clear()

app = FastAPI(title="Aerospace Predictive Maintenance API", version="1.0", lifespan=lifespan)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnginePayload(BaseModel):
    engine_id: int
    rf_features: list[float]
    lstm_sequence: list[list[float]]

@app.post("/predict")
def predict_engine_status(payload: EnginePayload, db: Session = Depends(get_db)):
    """
    Receives engine telemetry, calculates RUL, detects anomalies, and queries RAG if failing.
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

        rag_context = None
        if is_anomaly == "ANOMALY":
            query_text = f"Engine {payload.engine_id} showing severe sensor deviation. High reconstruction error indicating critical failure."
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
        
        #  Log to PostgreSQL
        return {
            "prediction_id": new_log.id,
            "engine_id": payload.engine_id,
            "status": "🚨 CRITICAL FAULT" if is_anomaly == "ANOMALY" else "✅ HEALTHY",
            "predicted_rul_cycles": round(predicted_rul, 2),
            "anomaly_mse": round(mse, 4),
            "anomaly_threshold": round(ANOMALY_THRESHOLD, 4),
            "maintenance_recommendation": rag_context
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/history/{engine_id}")
def get_engine_history(engine_id: int, db: Session = Depends(get_db)):
    """
    Retrieves the prediction history for a specific engine from PostgreSQL,
    and formats it to match the standard API response.
    """
    records = db.query(models.PredictionLog).filter(models.PredictionLog.engine_id == engine_id).order_by(models.PredictionLog.timestamp.desc()).limit(10).all()
    if not records:
        raise HTTPException(status_code=404, detail="No logs found for this engine.")
    
    # Translate the raw database rows into our frontend-friendly JSON format
    formatted_history = []
    for r in records:
        formatted_history.append({
            "prediction_id": r.id,
            "timestamp": r.timestamp,
            "status": "🚨 CRITICAL FAULT" if r.is_anomaly == "ANOMALY" else "✅ HEALTHY",
            "predicted_rul_cycles": round(r.predicted_rul, 2),
            "anomaly_mse": round(r.reconstruction_mse, 4),
            "anomaly_threshold": round(ANOMALY_THRESHOLD, 4),
            "maintenance_recommendation": r.rag_diagnostics
        })
        
    return formatted_history