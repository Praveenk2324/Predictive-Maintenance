import streamlit as st
import requests
import pandas as pd

# --- Configuration ---
API_URL = "http://api:8000"
st.set_page_config(page_title="Aerospace Maintenance", page_icon="✈️", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/features_train.parquet")

df = load_data()
engine_ids = df['engine_id'].unique().tolist()

# --- UI Sidebar ---
st.sidebar.title("✈️ Fleet Control")
st.sidebar.markdown("Select an engine and a specific flight cycle to simulate streaming telemetry.")

selected_engine = st.sidebar.selectbox("Select Engine ID", engine_ids)

# --- NEW: Time Travel Slider ---
# Find out how long this specific engine lived
engine_full_history = df[df['engine_id'] == selected_engine].sort_values('cycle')
max_cycle = int(engine_full_history['cycle'].max())

# Create a slider so we can test the engine when it was young and healthy!
# Minimum is 30 because our LSTM requires exactly 30 cycles of history to work.
selected_cycle = st.sidebar.slider(
    "Current Flight Cycle (Time Travel)", 
    min_value=30, 
    max_value=max_cycle, 
    value=max_cycle
)

# --- UI Main Body ---
st.title("Aerospace Predictive Maintenance Dashboard")

if st.button(f"🔍 Run Diagnostics at Cycle {selected_cycle}", type="primary"):
    with st.spinner("Transmitting telemetry to AI API..."):
        
        # 1. Filter data up to the cycle the user selected on the slider
        historical_data = engine_full_history[engine_full_history['cycle'] <= selected_cycle]
        USEFUL_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
        
        # Grab the windows based on the simulated current time
        lstm_data = historical_data.iloc[-30:][USEFUL_SENSORS].values.tolist()
        rf_data = historical_data.iloc[-10:][USEFUL_SENSORS].values.flatten().tolist()
        
        payload = {
            "engine_id": int(selected_engine),
            "rf_features": rf_data,
            "lstm_sequence": lstm_data
        }
        
        # 2. Call the FastAPI endpoint
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 3. Display the Results beautifully
            st.markdown(f"### Live Prediction Results (Cycle {selected_cycle})")
            col1, col2, col3 = st.columns(3)
            
            # Status styling
            if "CRITICAL" in result["status"]:
                col1.error(result["status"])
                st.warning(f"**🛠️ Maintenance Recommendation (RAG System):**\n\n{result['maintenance_recommendation']}")
            else:
                col1.success(result["status"])
                st.info(f"**📝 System Log:**\n\n{result['maintenance_recommendation']}")
                
            col2.metric("Remaining Useful Life (Cycles)", result["predicted_rul_cycles"])
            
            # NEW: Display MSE and the Threshold clearly
            col3.metric("Anomaly Score (MSE)", result["anomaly_mse"])
            col3.caption(f"🚨 **Threshold Limit:** {result.get('anomaly_threshold', 'N/A')}")

        except Exception as e:
            st.error(f"API Connection Failed: Ensure FastAPI is running on port 8000. Error: {e}")

st.markdown("---")

# --- Database History Section ---
st.subheader("🗄️ PostgreSQL Maintenance Logs")
if st.button("Fetch Historical Logs"):
    with st.spinner("Querying Database..."):
        try:
            history_res = requests.get(f"{API_URL}/history/{selected_engine}")
            if history_res.status_code == 200:
                history_data = history_res.json()
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True)
                else:
                    st.info("No logs found for this engine.")
            else:
                st.warning(f"Engine {selected_engine} has no historical data logged yet.")
        except Exception as e:
            st.error(f"Failed to fetch history: {e}")