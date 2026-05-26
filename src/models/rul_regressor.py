import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- Configurations ---
TRAIN_PATH = "data/processed/features_train.parquet"
TEST_PATH = "data/processed/features_test.parquet"
MLFLOW_URI = "http://127.0.0.1:5000"

def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    drop_cols = ['engine_id', 'cycle', 'cycle_norm', 'RUL', 'setting_1', 'setting_2', 'setting_3']
    features  = [col for col in train_df.columns if col not in drop_cols]

    X_train, y_train = train_df[features], train_df['RUL']
    X_test, y_test = test_df[features], test_df['RUL']

    return X_train, y_train, X_test, y_test, features

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 250),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
        'random_state': 42,
        'n_jobs': -1
    }

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    return np.sqrt(mean_squared_error(y_test, preds))

def main():
    print("Loading processed parquet data...")
    X_train, y_train, X_test, y_test, feature_names = load_data()

    print(f"Training on {len(feature_names)} features...")
    print("Starting Optuna Hyperparameter Tuning (50 trials)...")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction = 'minimize',
        sampler = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)

    print(f"Best Optuna Trial RMSE: {study.best_trial.value:.4f}")

    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    print("Training final model and logging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment('NASA_CMAPSS_RUL')

    with mlflow.start_run(run_name="RandomForest_Optuna_Best"):
        #Train final model
        rf_final = RandomForestRegressor(**best_params)
        rf_final.fit(X_train, y_train)

        preds = rf_final.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_params(best_params)
        mlflow.log_metrics({'rmse': rmse, 'r2':r2})

        importances = rf_final.feature_importances_
        indices = np.argsort(importances)[-15:]

        plt.figure(figsize=(10, 6))
        plt.title("Top 15 Feature Importances (Random Forest)")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()        

        os.makedirs("artifacts", exist_ok=True)
        plot_path = "artifacts/features_importance.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        mlflow.sklearn.log_model(rf_final, "ruf_rf_model")

        print(f"Run complete! Metrics logged ton MLflow:")
        print(f"-> RMSE: {rmse:.4f}")
        print(f"-> R2:   {r2:.4f}")

if __name__ == "__main__":
    main()