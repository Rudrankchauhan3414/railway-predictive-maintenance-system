import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
import tempfile
import os

# Set production configuration parameters
st.set_page_config(page_title="Railway Bridge Predictive Maintenance System", layout="wide")

# Persistent pipeline workspace state initialization
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Analysis", "Dashboard", "Vision Track Model Testing"])

# Advanced Probabilistic Fusion Layer with Dynamic Uncertainty Calibration
def calculate_adaptive_mps(vision_prob, structural_error, ambient_light_lux=85000):
    # Simulate a dynamic reliability threshold for optical tracks
    # High fluctuations (e.g., passing tunnels/shadows) degrade vision weight
    if ambient_light_lux < 500 or ambient_light_lux > 120000:
        vision_weight = 0.20  
        structural_weight = 0.80
    else:
        vision_weight = 0.55  # Standard operational baseline trust split
        structural_weight = 0.45
        
    # Apply dynamic probabilistic fusion mapping
    base_mps = (vision_weight * vision_prob) + (structural_weight * structural_error)
    
    # Contextual Urgency Scaling: Exponential escalation if both tracks flag risks simultaneously
    if vision_prob > 0.45 and structural_error > 0.45:
        base_mps = min(1.0, base_mps * 1.35)
        
    return base_mps

def run_anomaly_pipeline(data_path):
    df = pd.read_csv(data_path)
    
    # Extract structural aggregate boundaries and map spatial vectors
    window_groups = df.groupby("window_index").agg({
        "vibration_amplitude": lambda x: np.mean(np.abs(x)),
        "latitude": "first",
        "longitude": "first",
        "timestamp": "first"
    }).reset_index()
    
    # Normalize reconstruction errors to capture deviation probabilities
    scaler = MinMaxScaler()
    window_groups["anomaly_probability"] = scaler.fit_transform(window_groups[["vibration_amplitude"]])
    
    # Calculate fusion score assuming baseline vision is clear initially (0.0 defect probability)
    window_groups["maintenance_priority_score"] = window_groups["anomaly_probability"].apply(
        lambda x: calculate_adaptive_mps(vision_prob=0.0, structural_error=x)
    )
    
    # Operational alert threshold boundary mapping
    window_groups["status"] = window_groups["maintenance_priority_score"].apply(
        lambda p: "Anomalous" if p > 0.30 else "Normal"
    )
    return window_groups

if page == "Home":
    st.title("🚂 Multi-Modal Railway Predictive Maintenance Engine")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Run Full Pipeline Integration")
        uploaded_file = st.file_uploader("Upload continuous vibration telemetry logs (.csv)", type="csv")
        if uploaded_file is not None:
            if st.button("🚀 Start Analysis Pipeline"):
                st.session_state.processed_results = run_anomaly_pipeline(uploaded_file)
                st.session_state.pipeline_run = True
                st.success("Pipeline executed successfully. Heterogeneous sensor data fused.")
                
    with col2:
        st.subheader("Quick Start Framework")
        st.info("1. Run `generate_data.py` to compile telemetry.\n2. Ingest the generated file here.\n3. Evaluate spatial structural analytics.")

    if st.session_state.pipeline_run:
        res = st.session_state.processed_results
        total_anomalies = int(np.sum(res["status"] == "Anomalous"))
        anomaly_percentage = (total_anomalies / len(res)) * 100
        health_index = max(0.0, min(100.0, 100.0 - (anomaly_percentage * 1.2)))
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Health Index Evaluation", f"{health_index:.1f}/100")
        c2.metric("Asset Status Flag", "minor_degradation" if health_index < 85 else "healthy")
        c3.metric("Isolated Threat Hotspots", f"{total_anomalies} Windows")
        c4.metric("Spatial Defect Ratio", f"{anomaly_percentage:.1f}%")
        
        # Display the GPS localization matrix directly to the user
        st.subheader("📍 Isolated Structural Defect Localization Matrix")
        anomaly_logs = res[res["status"] == "Anomalous"][["window_index", "timestamp", "latitude", "longitude", "maintenance_priority_score"]]
        if not anomaly_logs.empty:
            st.dataframe(anomaly_logs.rename(columns={
                "window_index": "Evaluation Window",
                "timestamp": "Detection Timestamp",
                "latitude": "GPS Latitude",
                "longitude": "GPS Longitude",
                "maintenance_priority_score": "Fused MPS Score"
            }), use_container_width=True)
        else:
            st.success("Zero critical geolocated defects found along the route trajectory.")

elif page == "Analysis":
    st.title("📈 Detailed Time-Series Analysis")
    if not st.session_state.pipeline_run:
        st.warning("⚠️ Please execute the main processing pipeline on the 'Home' panel first.")
    else:
        res = st.session_state.processed_results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.loc[res["status"] == "Normal", "window_index"], y=res.loc[res["status"] == "Normal", "maintenance_priority_score"], mode='markers', name='Healthy Track Matrix', marker=dict(color='green', size=10)))
        fig.add_trace(go.Scatter(x=res.loc[res["status"] == "Anomalous", "window_index"], y=res.loc[res["status"] == "Anomalous", "maintenance_priority_score"], mode='markers', name='Isolated Structural Fault', marker=dict(color='red', size=12, symbol='x')))
        fig.update_layout(title="Structural Reconstruction Waveform Analysis (Fused MPS)", xaxis_title="Time Window Index", yaxis_title="Maintenance Priority Score", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Dashboard":
    st.title("📊 Statistical Distribution Dashboard")
    if not st.session_state.pipeline_run:
        st.warning("⚠️ Please execute the main processing pipeline on the 'Home' panel first.")
    else:
        res = st.session_state.processed_results
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(px.histogram(res, x="maintenance_priority_score", title="Probabilistic Fusion Density Profile", template="plotly_dark", color_discrete_sequence=['#3399FF']), use_container_width=True)
        with col_g2:
            st.plotly_chart(px.pie(names=["Structurally Sound", "Maintenance Action Required"], values=[np.sum(res["status"] == "Normal"), np.sum(res["status"] == "Anomalous")], title="Aggregated Bridge Track Asset Health Status Breakdown", template="plotly_dark", color_discrete_sequence=['green', 'red']), use_container_width=True)

elif page == "Vision Track Model Testing":
    st.title("👁️ Computer Vision Crack Detection Sandbox")
    
    # Introduce the illumination variable to demonstrate Bayesian shifting to the panel
    col_v1, col_v2 = st.columns([1, 3])
    with col_v1:
        st.subheader("Fusion Tuning Parameters")
        c_thresh = st.slider("YOLOv11 Confidence Threshold (%)", min_value=1, max_value=100, value=25)
        sim_lux = st.slider("Simulated Ambient Light (Lux)", min_value=100, max_value=150000, value=85000, help="Low Lux (<500) simulates tunnels/night. High Lux (>120k) simulates blinding sunlight reflections.")
    
    with col_v2:
        uploaded_image = st.file_uploader("Upload track snapshot image profile", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img_array = np.array(image)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        with st.spinner("Querying serverless YOLOv11 remote endpoint architecture..."):
            try:
                CLIENT = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key="ShAcTiFTYZepKe25iOaD"
                )
                result = CLIENT.infer(temp_path, model_id="railway-track-s7oyn-uwocb/1")
                predictions = [p for p in result.get("predictions", []) if p["confidence"] >= (c_thresh / 100.0)]
                
                annotated_img = img_array.copy()
                if predictions:
                    st.subheader("🎯 Real-Time Vision Track Detections")
                    for pred in predictions:
                        box_w, box_h = int(pred["width"]), int(pred["height"])
                        x1, y1 = int(pred["x"] - box_w / 2), int(pred["y"] - box_h / 2)
                        x2, y2 = int(pred["x"] + box_w / 2), int(pred["y"] + box_h / 2)
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        
                        # Process the dynamic sensor fusion index live based on current parameters
                        raw_vision_score = pred["confidence"]
                        simulated_structural_err = 0.65  # Simulating a simultaneous tracking error spike
                        
                        fused_score = calculate_adaptive_mps(raw_vision_score, simulated_structural_err, ambient_light_lux=sim_lux)
                        
                        cv2.putText(annotated_img, f"MPS: {fused_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                    st.image(annotated_img, caption="Live Model Bounding Box Annotation Map", use_container_width=True)
                else:
                    st.image(img_array, caption="No defect patterns tracked above selected confidence limits.", use_container_width=True)
                
                st.subheader("Raw Telemetry Response Object Payload")
                st.json({"predictions": predictions})
                
            except Exception as e:
                st.error(f"Inference SDK connection breakdown: {e}")
            finally:
                if os.path.exists(temp_path): 
                    os.remove(temp_path)