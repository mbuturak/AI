import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Başlık
st.title("YOLOv8 ile Nesne Algılama")
st.sidebar.title("Proje Ayarları")

# Modeli varsayılan olarak yükleme
model_path = "weights/best.pt"

try:
    model = YOLO(model_path)
    st.sidebar.success("Model başarıyla yüklendi!")
except Exception as e:
    st.sidebar.error(f"Model yüklenemedi: {e}")
    st.stop()

# Görsel yükleme
uploaded_file = st.file_uploader("Bir Görsel Yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli yükle ve göster
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Algılama işlemi
    if st.button("Algılama Yap"):
        with st.spinner("Algılama yapılıyor..."):
            img_array = np.array(image)
            results = model(img_array)

            # Algılama sonuçlarını çiz
            annotated_image = results[0].plot()
            st.image(annotated_image, caption="Algılama Sonuçları", use_column_width=True)
