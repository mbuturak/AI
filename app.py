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

        # Renk paleti tanımlama
        colors = [
            '#FF0000',  # Kırmızı
            '#FFA500',  # Turuncu
            '#FFFF00',  # Sarı
            '#00FF00',  # Yeşil
            '#00FFFF',  # Cyan
            '#0000FF',  # Mavi
            '#800080'   # Mor
        ]

        # Add boxes with hover information
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)
            label = results[0].names[cls]
            
            # Renk seçimi
            color = colors[i % len(colors)]
            
            # Köşe noktaları için kontrol noktaları hesaplama
            cp_x = (x2 - x1) * 0.2  # kontrol noktası offset'i
            cp_y = (y2 - y1) * 0.2

            # Bezier eğrisi için path
            path = f"M {x1},{y1} Q {(x1+x2)/2},{y1} {x2},{y1} Q {x2},{(y1+y2)/2} {x2},{y2} Q {(x1+x2)/2},{y2} {x1},{y2} Q {x1},{(y1+y2)/2} {x1},{y1}"
            
            # Add shape with curved edges
            fig.add_shape(
                type="path",
                path=path,
                line=dict(
                    color=color,
                    width=3,
                ),
                fillcolor="rgba(0,0,0,0)",
                opacity=0.7
            )
            
            # Add hover annotation
            fig.add_trace(go.Scatter(
                x=[(x1 + x2)/2],
                y=[(y1 + y2)/2],
                mode="markers",
                marker=dict(size=1, color=color),
                hoverinfo="text",
                text=f"{label}<br>Confidence: {conf:.2f}",
                showlegend=False
            ))

        # Update layout
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='black',
            paper_bgcolor='black',
            width=800,
            height=600
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
