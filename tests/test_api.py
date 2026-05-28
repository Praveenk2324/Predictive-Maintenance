import pandas as pd
import requests
import json

def test_prediction_api():
    print("Loading test data...")
    # Load the training data so we can grab an engine that is actually failing
    df = pd.read_parquet("data/processed/features_train.parquet")
    
    # Grab Engine 1
    engine_id = 62
    engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
    
    # The exact 14 sensors our models were trained on
    USEFUL_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    # 1. Prepare LSTM Data (The last 30 cycles of the engine's life)
    lstm_data = engine_data.iloc[-30:][USEFUL_SENSORS].values.tolist()
    
    # 2. Prepare RF Data (A flattened 10-cycle window: 10 x 14 = 140 features)
    rf_data = engine_data.iloc[-10:][USEFUL_SENSORS].values.flatten().tolist()
    
    # 3. Build the Payload
    payload = {
        "engine_id": engine_id,
        "rf_features": rf_data,
        "lstm_sequence": lstm_data
    }
    
    # 4. Send the Request to the FastAPI Server
    print("Sending request to FastAPI...")
    url = "http://127.0.0.1:8000/predict"
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise an error if the API crashes
        
        print("\n--- API RESPONSE ---")
        print(json.dumps(response.json(), indent=4))
        
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        if 'response' in locals() and response is not None:
            print(response.text)

if __name__ == "__main__":
    test_prediction_api()