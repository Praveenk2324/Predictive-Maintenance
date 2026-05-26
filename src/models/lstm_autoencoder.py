import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import joblib
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = 'data/processed/features_train.parquet'
MLFLOW_URI = 'http://127.0.0.1:5000'
USEFUL_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
HIDDEN_DIM = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim, seq_len):
        super(LSTMAutoencoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)

        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)
    
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)

        hidden = hidden[-1]

        repeated_hidden = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        decoder_out, _ = self.decoder(repeated_hidden)
        out = self.output_layer(decoder_out)

        return out

def create_sequences(df, seq_len, feature_cols, healthy_threshold=None):
    X = []
    for engine_id, group in df.groupby('engine_id'):
        group = group.sort_values('cycle')
        group_data = group[feature_cols].values
        group_rul = group['RUL'].values

        for i in range(len(group) - seq_len + 1):
            window_data = group_data[i : i + seq_len]
            window_rul = group_rul[i + seq_len - 1]

            if healthy_threshold is None or window_rul > healthy_threshold:
                X.append(window_data)

    return np.array(X)

def main():
    print(f"Using device: {device}")

    print("Loading processed parquet data...")
    train_df = pd.read_parquet(TRAIN_PATH)

    scaler = StandardScaler()
    train_df[USEFUL_SENSORS] = scaler.fit_transform(train_df[USEFUL_SENSORS])

    os.makedirs("artifacts", exist_ok=True)
    scaler_path = "artifacts/sensor_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print("Generating heallthy sequences (RUL > 80)...")
    X_train_healthy = create_sequences(train_df, SEQ_LEN, USEFUL_SENSORS, healthy_threshold=80)
    print(f"Healthy sequence tensor shape: {X_train_healthy.shape}")

    tensor_x = torch.tensor(X_train_healthy, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_x)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMAutoencoder(n_features=len(USEFUL_SENSORS), hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting MLflow run and training model...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("NASA_CMAPSS_Anomaly")

    with mlflow.start_run(run_name="LSTM_Autoencoder"):
        mlflow.log_params({"seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR, "hidden_dim": HIDDEN_DIM})

        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

    print("Computing anomaly threshold...")
    model.eval()
    with torch.no_grad():
        tensor_x_device = tensor_x.to(device)
        reconstruction = model(tensor_x_device)
        mse_per_sequence = torch.mean((reconstruction - tensor_x_device) ** 2, dim=(1, 2)).cpu().numpy()
    
    anomaly_threshold = np.percentile(mse_per_sequence, 95)
    print(f"Calculated 95th Percentile Anomaly Threshold: {anomaly_threshold:.6f}")
    mlflow.log_metric("anomaly_threshold", anomaly_threshold)

    plt.figure(figsize=(10, 6))
    plt.hist(mse_per_sequence, bins=50, alpha=0.7, color='blue')
    plt.axvline(anomaly_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (95%): {anomaly_threshold:.4f}')
    plt.title('Distribution of Reconstruction Errors (Healthy Data)')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()

    plot_path = 'artifacts/reconstruction_error_dist.png'
    plt.savefig(plot_path)

    mlflow.log_artifact(plot_path)
    mlflow.log_artifact(scaler_path)
    mlflow.pytorch.log_model(model, "lstm_autoencoder_model")
        
    print("Training complete! Model and artifacts logged to MLflow.")

if __name__ == "__main__":
    main()