import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
from pathlib import Path
import warnings

# Uyarıları gizle
warnings.filterwarnings('ignore')

# Başlık
st.title("Nesne Algılama")
st.sidebar.title("Proje Ayarları")

# Modeli varsayılan olarak yükleme
current_dir = Path(__file__).parent
model_path = str(current_dir / "weights" / "best.pt")

try:
    model = YOLO(model_path)
    st.sidebar.success("Model başarıyla yüklendi!")
except Exception as e:
    # st.sidebar.error(f"Model yüklenemedi: {e}")
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
        
        # Sadece etiketlenmiş sonucu göster
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Algılama Sonuçları", use_column_width=True)
