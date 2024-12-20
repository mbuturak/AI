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
    model = YOLO("best.pt")  # Update with your model path
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

        # Add image
        fig.add_trace(go.Image(z=img))

        # Define color palette
        colors = [
            'rgba(255, 0, 0, 0.5)',  # Red
            'rgba(255, 165, 0, 0.5)',  # Orange
            'rgba(255, 255, 0, 0.5)',  # Yellow
            'rgba(0, 255, 0, 0.5)',  # Green
            'rgba(0, 255, 255, 0.5)',  # Cyan
            'rgba(0, 0, 255, 0.5)',  # Blue
            'rgba(128, 0, 128, 0.5)'   # Purple
        ]

        # Add detected areas with curved edges
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)
            label = results[0].names[cls]
            
            # Select color
            color = colors[i % len(colors)]
            
            # Create a smooth oval-like path for the box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # Center point
            rx, ry = (x2 - x1) / 2, (y2 - y1) / 2  # Radii for x and y

            # Define SVG path for an ellipse
            path = (
                f"M {cx - rx},{cy} "
                f"A {rx},{ry} 0 1,1 {cx + rx},{cy} "
                f"A {rx},{ry} 0 1,1 {cx - rx},{cy}"
            )
            
            # Add shape with curved edges
            fig.add_shape(
                type="path",
                path=path,
                line=dict(
                    color=color,
                    width=3,
                ),
                fillcolor=color,
                opacity=0.3
            )
            
            # Add hover annotation
            fig.add_trace(go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                marker=dict(size=5, color=color),
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
