import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

from data_loader import load_cmaps_data
from preprocessing import (
    add_rul_column, get_constant_sensors, remove_constant_sensors,
    create_anomaly_labels, prepare_features
)

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Deep Space Mission Control",
    page_icon="🛰️",
    layout="wide"
)

# Title
st.title("🛰️ Deep Space Mission Control - AI Anomaly Detection System")
st.markdown("---")

# Sidebar
st.sidebar.header("Mission Parameters")
engine_id = st.sidebar.selectbox("Select Engine/Subsystem", range(1, 11))
alert_threshold = st.sidebar.slider("Alert Sensitivity", 0.0, 1.0, 0.5, 0.05)

@st.cache_data
def load_and_process():
    """Load and process data with caching."""
    train_df, test_df, rul_test = load_cmaps_data()
    
    # Process training data
    train_df = add_rul_column(train_df)
    constant_sensors = get_constant_sensors(train_df)
    train_df = remove_constant_sensors(train_df, constant_sensors)
    train_df = create_anomaly_labels(train_df, rul_threshold=30)
    
    return train_df, constant_sensors

@st.cache_resource
def train_model():
    """Train isolation forest model."""
    train_df, _ = load_and_process()
    
    feature_cols = [col for col in train_df.columns if col not in 
                    ['unit_number', 'time_in_cycles', 'RUL', 'is_anomaly']]
    X = train_df[feature_cols].values
    y = train_df['is_anomaly'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(contamination=y.mean(), random_state=42)
    model.fit(X_scaled)
    
    return model, scaler, feature_cols

def get_anomaly_score(model, scaler, data_point, feature_cols):
    """Get anomaly score for a single data point."""
    data_scaled = scaler.transform(data_point.reshape(1, -1))
    score = model.decision_function(data_scaled)[0]
    return score

def get_decision(score, threshold):
    """Make autonomous decision based on score."""
    if score < -0.3:
        return "🚨 CRITICAL", "Immediate shutdown - Switching to redundant system", "red"
    elif score < -0.1:
        return "⚠️ WARNING", "Isolate subsystem - Alert ground station", "orange"
    else:
        return "✅ NORMAL", "Continue nominal operations", "green"

# Load data
with st.spinner("Loading telemetry data..."):
    train_df, constant_sensors = load_and_process()
    model, scaler, feature_cols = train_model()

# Filter for selected engine
engine_data = train_df[train_df['unit_number'] == engine_id]

# Main dashboard - 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Subsystem", f"Engine {engine_id}")
    st.metric("Total Cycles", f"{len(engine_data)}")
    st.metric("Anomalies Detected", f"{engine_data['is_anomaly'].sum()}")

with col2:
    rul_current = engine_data['RUL'].iloc[-1] if len(engine_data) > 0 else 0
    st.metric("Remaining Useful Life", f"{rul_current} cycles")
    
    anomaly_pct = (engine_data['is_anomaly'].sum() / len(engine_data) * 100) if len(engine_data) > 0 else 0
    st.metric("Anomaly Rate", f"{anomaly_pct:.1f}%")

with col3:
    # Get latest data point
    latest = engine_data.iloc[-1] if len(engine_data) > 0 else None
    if latest is not None:
        latest_features = latest[feature_cols].values
        anomaly_score = get_anomaly_score(model, scaler, latest_features, feature_cols)
        decision, action, color = get_decision(anomaly_score, alert_threshold)
        st.markdown(f"### Current Status")
        st.markdown(f"<h2 style='color:{color}'>{decision}</h2>", unsafe_allow_html=True)
        st.info(action)

st.markdown("---")

# Sensor plots
st.subheader("📊 Sensor Telemetry")

# Select sensors to display
available_sensors = [col for col in feature_cols if col.startswith('sensor_')]
selected_sensors = st.multiselect(
    "Select sensors to display",
    available_sensors,
    default=available_sensors[:3] if len(available_sensors) >= 3 else available_sensors
)

if selected_sensors:
    fig = make_subplots(
        rows=len(selected_sensors), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=selected_sensors
    )
    
    for i, sensor in enumerate(selected_sensors, 1):
        fig.add_trace(
            go.Scatter(
                x=engine_data['time_in_cycles'],
                y=engine_data[sensor],
                mode='lines',
                name=sensor,
                line=dict(color='blue', width=1)
            ),
            row=i, col=1
        )
        
        # Add anomaly highlights
        anomalies = engine_data[engine_data['is_anomaly'] == 1]
        if len(anomalies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomalies['time_in_cycles'],
                    y=anomalies[sensor],
                    mode='markers',
                    name=f'{sensor} (Anomaly)',
                    marker=dict(color='red', size=8, symbol='x'),
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
        
        fig.update_yaxes(title_text="Value", row=i, col=1)
    
    fig.update_xaxes(title_text="Time Cycles", row=len(selected_sensors), col=1)
    fig.update_layout(height=300 * len(selected_sensors), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# Anomaly timeline
st.subheader("🚨 Anomaly Timeline")
anomaly_events = engine_data[engine_data['is_anomaly'] == 1]
if len(anomaly_events) > 0:
    anomaly_df = anomaly_events[['time_in_cycles', 'RUL'] + selected_sensors[:3] if selected_sensors else []]
    st.dataframe(anomaly_df, use_container_width=True)
else:
    st.success("No anomalies detected in this subsystem")

# Decision log
st.subheader("📋 Autonomous Decision Log")
st.markdown("""
**Decision Logic:**
- **Score > -0.1**: NORMAL → Continue operations
- **Score -0.3 to -0.1**: WARNING → Isolate subsystem, alert ground station  
- **Score < -0.3**: CRITICAL → Immediate shutdown, switch to redundancy
""")

# Footer
st.markdown("---")
st.caption("India Space Academy | AI-Based Real-Time Anomaly Detection for Deep Space Missions")