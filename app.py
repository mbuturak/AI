import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
from pathlib import Path
import warnings
import plotly.express as px
import pandas as pd

# Uyarıları gizle
warnings.filterwarnings('ignore')

# Başlık
st.title("YOLOv8 ile Nesne Algılama")
st.sidebar.title("Proje Ayarları")

# Modeli varsayılan olarak yükleme
current_dir = Path(__file__).parent
model_path = str(current_dir / "weights" / "best.pt")

try:
    model = YOLO(model_path)
    st.sidebar.success("Model başarıyla yüklendi!")
except Exception as e:
    st.sidebar.error(f"Model yüklenemedi: {e}")
    st.stop()

# Görsel yükleme
uploaded_file = st.file_uploader("Bir Görsel Yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli yükle
    image = Image.open(uploaded_file)
    
    # Otomatik olarak algılama yap
    with st.spinner("Algılama yapılıyor..."):
        img_array = np.array(image)
        results = model(img_array)
        
        # Sonuçları göster
        col1, col2 = st.columns(2)
        
        with col1:
            # Etiketlenmiş görsel
            annotated_image = results[0].plot()
            st.image(annotated_image, caption="Algılama Sonuçları", use_container_width=True)
        
        with col2:
            # İstatistikler
            boxes = results[0].boxes
            if len(boxes) > 0:
                # Tespit edilen nesnelerin sayısı
                st.subheader("Tespit İstatistikleri")
                
                # Sınıf sayılarını hesapla
                classes = boxes.cls.cpu().numpy()
                names = results[0].names
                class_counts = {names[int(c)]: np.sum(classes == c) for c in np.unique(classes)}
                
                # Güven skorları
                confidences = boxes.conf.cpu().numpy()
                
                # Sınıf dağılımı pasta grafiği
                fig_pie = px.pie(
                    values=list(class_counts.values()),
                    names=list(class_counts.keys()),
                    title="Tespit Edilen Nesnelerin Dağılımı"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Güven skorları histogramı
                fig_hist = px.histogram(
                    confidences,
                    title="Güven Skorları Dağılımı",
                    labels={'value': 'Güven Skoru', 'count': 'Sayı'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Özet istatistikler
                st.subheader("Özet Bilgiler")
                st.write(f"🎯 Toplam tespit edilen nesne sayısı: {len(boxes)}")
                st.write(f"📊 Ortalama güven skoru: {confidences.mean():.2f}")
                st.write(f"🏆 En yüksek güven skoru: {confidences.max():.2f}")
                
                # Detaylı sınıf bilgileri
                st.subheader("Sınıf Detayları")
                for cls_name, count in class_counts.items():
                    st.write(f"📌 {cls_name}: {count} adet")
            else:
                st.warning("Görüntüde hiçbir nesne tespit edilemedi.")
