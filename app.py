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

st.set_page_config(page_title="Railway Bridge Predictive Maintenance System", layout="wide")

if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Analysis", "Dashboard", "Vision Track Model Testing"])

def run_anomaly_pipeline(data_path):
    df = pd.read_csv(data_path)
    window_groups = df.groupby("window_index")["vibration_amplitude"].apply(lambda x: np.mean(np.abs(x))).reset_index()
    scaler = MinMaxScaler()
    window_groups["anomaly_probability"] = scaler.fit_transform(window_groups[["vibration_amplitude"]])
    window_groups["status"] = window_groups["anomaly_probability"].apply(lambda p: "Anomalous" if p > 0.45 else "Normal")
    return window_groups

if page == "Home":
    st.title("🚂 Railway Bridge Predictive Maintenance System")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Run Full Pipeline Integration")
        uploaded_file = st.file_uploader("Upload continuous vibration telemetry logs (.csv)", type="csv")
        if uploaded_file is not None:
            if st.button("🚀 Start Analysis Pipeline"):
                st.session_state.processed_results = run_anomaly_pipeline(uploaded_file)
                st.session_state.pipeline_run = True
                st.success("Pipeline completed successfully!")
    with col2:
        st.subheader("Quick Start Framework")
        st.info("1. Run `generate_data.py` locally.\n2. Upload the file here.\n3. Observe system diagnostics.")

    if st.session_state.pipeline_run:
        res = st.session_state.processed_results
        total_anomalies = int(np.sum(res["status"] == "Anomalous"))
        anomaly_percentage = (total_anomalies / len(res)) * 100
        health_index = max(0.0, min(100.0, 100.0 - (anomaly_percentage * 1.2)))
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Health Index Evaluation", f"{health_index:.1f}/100")
        c2.metric("Asset Status", "minor_degradation" if health_index < 85 else "healthy")
        c3.metric("Total Anomalies Found", f"{total_anomalies}")
        c4.metric("Anomaly Ratio", f"{anomaly_percentage:.1f}%")

elif page == "Analysis":
    st.title("📈 Detailed Time-Series Analysis")
    if not st.session_state.pipeline_run:
        st.warning("⚠️ Please execute the main processing pipeline on the 'Home' panel first.")
    else:
        res = st.session_state.processed_results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.loc[res["status"] == "Normal", "window_index"], y=res.loc[res["status"] == "Normal", "anomaly_probability"], mode='markers', name='Normal State', marker=dict(color='green')))
        fig.add_trace(go.Scatter(x=res.loc[res["status"] == "Anomalous", "window_index"], y=res.loc[res["status"] == "Anomalous", "anomaly_probability"], mode='markers', name='Anomaly Spike', marker=dict(color='red')))
        fig.update_layout(title="Structural Reconstruction Analysis", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Dashboard":
    st.title("📊 Statistical Distribution Dashboard")
    if not st.session_state.pipeline_run:
        st.warning("⚠️ Please execute the main processing pipeline on the 'Home' panel first.")
    else:
        res = st.session_state.processed_results
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(px.histogram(res, x="anomaly_probability", title="Probability Density Profile", template="plotly_dark"), use_container_width=True)
        with col_g2:
            st.plotly_chart(px.pie(names=["Healthy", "Degradation"], values=[np.sum(res["status"] == "Normal"), np.sum(res["status"] == "Anomalous")], title="Track Health Status", template="plotly_dark"), use_container_width=True)

elif page == "Vision Track Model Testing":
    st.title("👁️ Computer Vision Crack Detection Sandbox")
    c_thresh = st.slider("Confidence Cutoff Threshold (%)", min_value=1, max_value=100, value=25)
    uploaded_image = st.file_uploader("Upload track snapshot", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img_array = np.array(image)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        with st.spinner("Querying serverless YOLOv11 remote endpoint..."):
            try:
                CLIENT = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key="YOUR_ACTUAL_ROBOFLOW_API_KEY"
                )
                result = CLIENT.infer(temp_path, model_id="railway-track-s7oyn-uwocb/1")
                predictions = [p for p in result.get("predictions", []) if p["confidence"] >= (c_thresh / 100.0)]
                
                annotated_img = img_array.copy()
                if predictions:
                    for pred in predictions:
                        box_w, box_h = int(pred["width"]), int(pred["height"])
                        x1, y1 = int(pred["x"] - box_w / 2), int(pred["y"] - box_h / 2)
                        x2, y2 = int(pred["x"] + box_w / 2), int(pred["y"] + box_h / 2)
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    st.image(annotated_img, caption="Live Detections Output Map", use_container_width=True)
                else:
                    st.image(img_array, caption="No detections found above threshold.", use_container_width=True)
                st.json({"predictions": predictions})
            except Exception as e:
                st.error(f"Inference failure: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)