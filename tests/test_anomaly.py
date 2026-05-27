import pandas as pd
import numpy as np
import torch
import joblib
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# FIX 1: Change to the training dataset path
TRAIN_PATH = "data/processed/features_train.parquet"
MLFLOW_URI = "http://127.0.0.1:5000"
USEFUL_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
SEQ_LEN = 30

def main():
    print("Connecting to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name("NASA_CMAPSS_Anomaly")
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    threshold = latest_run.data.metrics['anomaly_threshold']

    print(f"Loaded Run ID: {run_id}")
    print(f"Anomaly Threshold: {threshold:.6f}\n")

    scaler_path = client.download_artifacts(run_id, 'sensor_scaler.pkl')
    scaler = joblib.load(scaler_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_uri = f"runs:/{run_id}/lstm_autoencoder_model"
    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    model.eval()

    # FIX 2: Load the training data instead of test data
    train_df = pd.read_parquet(TRAIN_PATH)
    engine_1 = train_df[train_df['engine_id'] == 1].sort_values('cycle')

    healthy_data = engine_1.iloc[:SEQ_LEN][USEFUL_SENSORS]
    
    # In the training set, the last 30 cycles represent the engine actually dying (RUL counting down to 0)
    failing_data = engine_1.iloc[-SEQ_LEN:][USEFUL_SENSORS]  
    
    healthy_scaled = scaler.transform(healthy_data)
    failing_scaled = scaler.transform(failing_data)

    t_healthy = torch.tensor(healthy_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    t_failing = torch.tensor(failing_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    print("Running sequences through the LSTM Autoencoder...\n")   

    with torch.no_grad():
        rec_healthy = model(t_healthy)
        mse_healthy = torch.mean((rec_healthy - t_healthy) ** 2).item()

        rec_failing = model(t_failing)
        mse_failing = torch.mean((rec_failing - t_failing) ** 2).item()

    print("-" * 40)
    print("         ANOMALY DETECTION RESULTS       ")
    print("-" * 40)
    print(f"Healthy Engine MSE : {mse_healthy:.6f}")
    print(f"Status             : {'🚨 FAULT DETECTED' if mse_healthy > threshold else '✅ NORMAL'}\n")
    
    print(f"Failing Engine MSE : {mse_failing:.6f}")
    print(f"Status             : {'🚨 FAULT DETECTED' if mse_failing > threshold else '✅ NORMAL'}")
    print("-" * 40)

if __name__ == "__main__":
    main()