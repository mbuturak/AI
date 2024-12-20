import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
from pathlib import Path
import warnings
import plotly.express as px
import pandas as pd

# Hide warnings
warnings.filterwarnings('ignore')

# Title
st.title("Object Detection with YOLOv8")
st.sidebar.title("Project Settings")

# Model configuration
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Load model
try:
    model = YOLO('yolov8n.pt')
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        
        # Perform detection automatically
        with st.spinner("Detecting objects..."):
            img_array = np.array(image)
            results = model(img_array, conf=confidence_threshold)
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                # Annotated image with labels
                annotated_image = results[0].plot(
                    boxes=True,       # Show boxes
                    labels=True,      # Show labels
                    conf=True,        # Show confidence scores
                    line_width=2,     # Line width
                    font_size=16      # Font size
                )
                st.image(annotated_image, caption="Detection Results", use_container_width=True)
                
                # Confidence score information
                st.info("""
                ### 📝 What is Confidence Score?
                Confidence score indicates how certain the model is about its detection, ranging from 0 to 1.
                
                #### 🎯 Confidence Score Ranges:
                - **0.90 - 1.00:** Excellent detection - Model is very confident
                - **0.70 - 0.89:** Good detection - Model is quite confident
                - **0.50 - 0.69:** Fair detection - Model is somewhat confident
                - **0.00 - 0.49:** Poor detection - Model is not confident
                
                #### 📊 Common Reasons for Low Confidence:
                - Blurry or dark image
                - Partial visibility of object
                - Poor lighting conditions
                - Object is too far away
                """)
            
            with col2:
                # Statistics
                if len(results[0].boxes) > 0:
                    # Number of detected objects
                    st.subheader("Detection Statistics")
                    
                    # Calculate class counts
                    boxes = results[0].boxes
                    classes = boxes.cls.cpu().numpy()
                    names = results[0].names
                    class_counts = {names[int(c)]: np.sum(classes == c) for c in np.unique(classes)}
                    
                    # Confidence scores
                    confidences = boxes.conf.cpu().numpy()
                    
                    # Class distribution pie chart
                    fig_pie = px.pie(
                        values=list(class_counts.values()),
                        names=list(class_counts.keys()),
                        title="Distribution of Detected Objects"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Confidence scores histogram
                    fig_hist = px.histogram(
                        confidences,
                        title="Distribution of Confidence Scores",
                        labels={'value': 'Confidence Score', 'count': 'Count'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Summary")
                    st.write(f"🎯 Total objects detected: {len(boxes)}")
                    st.write(f"📊 Average confidence score: {confidences.mean():.2f}")
                    st.write(f"🏆 Highest confidence score: {confidences.max():.2f}")
                    
                    # Detailed class information
                    st.subheader("Class Details")
                    for cls_name, count in class_counts.items():
                        st.write(f"📌 {cls_name}: {count} objects")
                else:
                    st.warning("No objects detected in the image.")
                    st.write("Try adjusting the confidence threshold in the sidebar.")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
