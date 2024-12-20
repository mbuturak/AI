import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import warnings
import plotly.graph_objects as go

# Hide warnings
warnings.filterwarnings('ignore')

# Title
st.title("Object Detection")
st.sidebar.title("Project Settings")

# Load model by default
try:
    model = YOLO("best.pt")
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
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

        # Add image as background
        fig.add_trace(go.Image(z=img))

        # Define color palette
        colors = [
            '#FF0000',  # Kırmızı
            '#FFA500',  # Turuncu
            '#FFFF00',  # Sarı
            '#00FF00',  # Yeşil
            '#00FFFF',  # Cyan
            '#0000FF',  # Mavi
            '#800080'   # Mor
        ]

        # Add detected areas with smooth shapes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Merkez ve yarıçap hesapla
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx, ry = (x2 - x1) / 2, (y2 - y1) / 2

            # Elips için noktalar oluştur
            t = np.linspace(0, 2*np.pi, 100)
            x = cx + rx * np.cos(t)
            y = cy + ry * np.sin(t)
            
            # Renk seç
            color = colors[i % len(colors)]
            
            # Yumuşak kenarlı elips ekle
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                fill="toself",
                fillcolor=color,
                line=dict(color=color),
                opacity=0.5,
                showlegend=False,
                hoverinfo='skip'
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
