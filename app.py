import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import warnings
import plotly.graph_objects as go
import cv2

# Hide warnings
warnings.filterwarnings('ignore')

# Title
st.title("Object Detection")
st.sidebar.title("Project Settings")

# Load model by default
try:
    model = YOLO("weights/best.pt", task='detect')
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Ensure image is in correct format
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        # Perform detection
        with st.spinner("Detecting objects..."):
            results = model.predict(img_array, conf=0.25)
            
            # Debug için sınıf isimlerini kontrol et
            st.write("Model Sınıfları:", results[0].names)
            
            # Create Plotly figure
            fig = go.Figure()

            # Add image as background
            fig.add_trace(go.Image(z=img_array))

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
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    cls = int(box.cls)
                    
                    # Sınıf ismini al ve kontrol et
                    try:
                        label = results[0].names[cls]
                        st.write(f"Tespit {i+1}: Sınıf {cls} -> {label}")
                    except:
                        label = f"Sınıf {cls}"
                        st.write(f"Hata: Sınıf {cls} için isim bulunamadı")
                    
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
                        hoverinfo='text',
                        hovertext=f"{label}<br>Güven: {conf:.2%}",
                        hoverlabel=dict(
                            bgcolor=color,
                            font=dict(color='white')
                        )
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
            
    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")
        st.error("Lütfen farklı bir görüntü deneyin veya sayfayı yenileyin.")
