import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
from pathlib import Path
import warnings
import plotly.graph_objects as go

# Hide warnings
warnings.filterwarnings('ignore')

# Title
st.title("Object Detection")
st.sidebar.title("Project Settings")

# Load model by default
current_dir = Path(__file__).parent
model_path = str(current_dir / "weights" / "best.pt")

try:
    model = YOLO(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    # st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    
    # Perform detection automatically
    with st.spinner("Detecting objects..."):
        img_array = np.array(image)
        results = model(img_array)
        
        # Get detection results
        boxes = results[0].boxes
        img = results[0].orig_img

        # Create Plotly figure
        fig = go.Figure()

        # Add image
        fig.add_trace(go.Image(z=img))

        # Add boxes with hover information
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf)
            cls = int(box.cls)
            label = results[0].names[cls]
            
            # Add shape
            fig.add_shape(
                type="rect",
                x0=x1, y0=y1, x1=x2, y1=y2,
                line=dict(color="red", width=2),
                fillcolor="rgba(255,0,0,0.2)",
                hoverinfo="text",
            )
            
            # Add hover annotation
            fig.add_trace(go.Scatter(
                x=[x1],
                y=[y1],
                mode="markers",
                marker=dict(size=1, color="red"),
                hoverinfo="text",
                text=f"Class: {label}<br>Confidence: {conf:.2f}",
                showlegend=False
            ))

        # Update layout
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
