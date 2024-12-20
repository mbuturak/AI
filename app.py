import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import warnings
import plotly.graph_objects as go
from skimage import measure

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
        results = model(img_array, retina_masks=True)  # retina_masks=True daha detaylı maskeler için
        
        # Get detection results
        masks = results[0].masks
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

        # Add detected areas with masks
        if masks is not None:
            for i, mask in enumerate(masks):
                # Maske verilerini numpy dizisine dönüştür
                mask_array = mask.data.cpu().numpy()[0]
                
                # Maskenin konturlarını bul
                contours = measure.find_contours(mask_array, 0.5)
                
                # Her kontur için
                for contour in contours:
                    # Kontur noktalarını Plotly formatına dönüştür
                    x_coords = contour[:, 1]
                    y_coords = contour[:, 0]
                    
                    # Renk seç
                    color = colors[i % len(colors)]
                    
                    # Kontur çizgisini ekle
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
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
