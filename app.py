import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import warnings
import plotly.graph_objects as go
import cv2
import os

# Hide warnings
warnings.filterwarnings('ignore')

# Title
st.title("Object Detection")
st.sidebar.title("Project Settings")

# Model yükleme fonksiyonu
def load_model(model_path):
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model dosyası bulunamadı: {model_path}")
        return None
    
    try:
        model = YOLO(model_path, task='detect')
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB cinsinden
        st.sidebar.success(f"Model başarıyla yüklendi! (Boyut: {file_size:.2f} MB)")
        return model
    except Exception as e:
        st.sidebar.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None

# Model yükleme
MODEL_PATH = "weights/best.pt"
model = load_model(MODEL_PATH)

if model is None:
    st.error("Model yüklenemedi. Lütfen model dosyasını kontrol edin.")
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
                    label = results[0].names[cls]
                    
                    # Merkez ve yarıçap hesapla
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    rx, ry = (x2 - x1) / 2, (y2 - y1) / 2

                    # Elips için noktalar oluştur
                    t = np.linspace(0, 2*np.pi, 100)
                    x = cx + rx * np.cos(t)
                    y = cy + ry * np.sin(t)
                    
                    # Renk seç
                    color = colors[i % len(colors)]
                    
                    # Rengin parlaklığını kontrol et ve yazı rengini belirle
                    # Sarı ve açık renkler için siyah, koyu renkler için beyaz yazı
                    text_color = 'black' if color in ['#FFFF00', '#00FFFF'] else 'white'
                    
                    # Yumuşak kenarlı elips ekle
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        fill="toself",
                        fillcolor=color,
                        line=dict(color=color),
                        opacity=0.5,
                        name=label,
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f"{label}<br>Güven: {conf:.2%}",
                        hoverlabel=dict(
                            bgcolor=color,
                            font=dict(color=text_color)
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
                height=600,
                hoverlabel=dict(
                    namelength=-1  # Tüm ismi göster
                ),
                hovermode='closest'
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Analiz bölümü
            st.markdown("---")  # Ayırıcı çizgi
            
            # İki kolon oluştur
            col1, col2 = st.columns(2)
            
            with col1:
                # Tespit edilen bölgelerin dağılımı
                st.subheader("Distribution of Detected Regions")
                
                # Bölgelerin sayısını hesapla
                class_counts = {}
                for box in boxes:
                    cls = int(box.cls)
                    label = results[0].names[cls]
                    class_counts[label] = class_counts.get(label, 0) + 1
                
                # Dağılım grafiği
                dist_fig = go.Figure(data=[
                    go.Bar(
                        x=list(class_counts.keys()),
                        y=list(class_counts.values()),
                        marker_color='#1f77b4'
                    )
                ])
                dist_fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(dist_fig, use_container_width=True)
                
                # Model performans metrikleri
                st.subheader("Model Performance Metrics")
                
                # Ortalama güven skoru
                avg_conf = sum(float(box.conf) for box in boxes) / len(boxes) if boxes else 0
                
                # Metrik göstergeleri
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric("Average Confidence", f"{avg_conf:.1%}")
                with col_metrics2:
                    st.metric("Total Detections", len(boxes))
            
            with col2:
                # Açıklamalar
                st.subheader("Analysis Explanation")
                
                # Dağılım açıklaması
                st.markdown("""
                **Distribution Analysis:**
                The bar chart shows the frequency of detected regions in the X-Ray image. 
                Each bar represents the number of occurrences for each detected anatomical structure, 
                providing insights into the distribution of different bone structures in the image.
                """)
                
                st.markdown("""
                **Performance Metrics:**
                - **Average Confidence**: Indicates the model's average confidence level across all detections. 
                  Higher values suggest stronger certainty in the predictions.
                - **Total Detections**: Shows the total number of regions identified in the image, 
                  helping to understand the complexity of the analysis.
                """)
                
                # Önemli notlar
                st.markdown("""
                **Note:** This analysis is based on AI-powered detection and should be used as 
                a supportive tool alongside professional medical evaluation. The confidence scores 
                and distributions provide additional context but should not be used as the sole 
                basis for medical decisions.
                """)

    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")
        st.error("Lütfen farklı bir görüntü deneyin veya sayfayı yenileyin.")
