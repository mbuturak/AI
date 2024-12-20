import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
from pathlib import Path
import warnings

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
        
        # Show only the annotated result
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Detection Results", use_container_width=True)
