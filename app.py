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

# Dil seçenekleri için metinler
TEXTS = {
    'English': {
        'title': "Object Detection",
        'settings': "Project Settings",
        'language': "Language",
        'upload': "Upload an Image",
        'loading': "Detecting objects...",
        'distribution_title': "Distribution of Detected Regions",
        'metrics_title': "Model Performance Metrics",
        'avg_confidence': "Average Confidence",
        'total_detections': "Total Detections",
        'analysis_title': "Analysis Explanation",
        'distribution_text': """
        **Distribution Analysis:**
        The visualizations show the distribution of detected regions in the X-Ray image:
        
        - **Bar Chart**: Shows the count of each detected region type, providing a clear 
        comparison of frequencies across different anatomical structures.
        
        - **Pie Chart**: Displays the percentage distribution of detected regions, helping 
        to understand the relative proportion of each anatomical structure in the image.
        
        This analysis helps in quickly identifying the prevalence and distribution patterns 
        of different bone structures in the X-Ray image.
        """,
        'metrics_text': """
        **Performance Metrics:**
        - **Average Confidence**: Indicates the model's average confidence level across all detections. 
          Higher values suggest stronger certainty in the predictions.
        - **Total Detections**: Shows the total number of regions identified in the image, 
          helping to understand the complexity of the analysis.
        """,
        'note_text': """
        **Note:** This analysis is based on AI-powered detection and should be used as 
        a supportive tool alongside professional medical evaluation. The distributions 
        provide additional context but should not be used as the sole basis for medical decisions.
        """
    },
    'Türkçe': {
        'title': "Nesne Tespiti",
        'settings': "Proje Ayarları",
        'language': "Dil",
        'upload': "Görüntü Yükle",
        'loading': "Nesneler tespit ediliyor...",
        'distribution_title': "Tespit Edilen Bölgelerin Dağılımı",
        'metrics_title': "Model Performans Metrikleri",
        'avg_confidence': "Ortalama Güven",
        'total_detections': "Toplam Tespit",
        'analysis_title': "Analiz Açıklaması",
        'distribution_text': """
        **Dağılım Analizi:**
        Grafikler, X-Ray görüntüsünde tespit edilen bölgelerin dağılımını göstermektedir:
        
        - **Çubuk Grafik**: Her tespit edilen bölge tipinin sayısını gösterir ve farklı 
        anatomik yapıların sıklığını karşılaştırmayı sağlar.
        
        - **Pasta Grafik**: Tespit edilen bölgelerin yüzdesel dağılımını gösterir ve 
        her anatomik yapının görüntüdeki göreceli oranını anlamamıza yardımcı olur.
        
        Bu analiz, X-Ray görüntüsündeki farklı kemik yapılarının yaygınlığını ve dağılım 
        modellerini hızlıca belirlemeye yardımcı olur.
        """,
        'metrics_text': """
        **Performans Metrikleri:**
        - **Ortalama Güven**: Tüm tespitler için modelin ortalama güven seviyesini gösterir. 
          Yüksek değerler, tahminlerde daha güçlü kesinlik olduğunu gösterir.
        - **Toplam Tespit**: Görüntüde tanımlanan toplam bölge sayısını gösterir ve 
          analizin karmaşıklığını anlamamıza yardımcı olur.
        """,
        'note_text': """
        **Not:** Bu analiz yapay zeka destekli tespit üzerine kuruludur ve profesyonel 
        tıbbi değerlendirme ile birlikte destekleyici bir araç olarak kullanılmalıdır. 
        Dağılımlar ek bağlam sağlar, ancak tek başına tıbbi kararlar için kullanılmamalıdır.
        """
    }
}

# Title
st.title("X-Ray Analysis")

# Sidebar
st.sidebar.title("Project Settings")

# Dil seçimi
selected_language = st.sidebar.selectbox(
    "Language / Dil",
    ["English", "Türkçe"]
)

# Seçilen dile göre metinleri al
texts = TEXTS[selected_language]

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
uploaded_file = st.file_uploader(texts['upload'], type=["jpg", "jpeg", "png"])

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
        with st.spinner(texts['loading']):
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
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(texts['distribution_title'])
                
                # Bölgelerin sayısını hesapla
                class_counts = {}
                for box in boxes:
                    cls = int(box.cls)
                    label = results[0].names[cls]
                    class_counts[label] = class_counts.get(label, 0) + 1
                
                # Bar grafiği
                dist_fig = go.Figure(data=[
                    go.Bar(
                        x=list(class_counts.keys()),
                        y=list(class_counts.values()),
                        marker_color='#1f77b4',
                        text=list(class_counts.values()),
                        textposition='auto'
                    )
                ])
                dist_fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    title=dict(
                        text="Bölge Sayıları" if selected_language == "Türkçe" else "Region Counts",
                        font=dict(color='white')
                    )
                )
                st.plotly_chart(dist_fig, use_container_width=True)

                # Pasta grafik
                pie_fig = go.Figure(data=[
                    go.Pie(
                        labels=list(class_counts.keys()),
                        values=list(class_counts.values()),
                        hole=.3,
                        textinfo='percent',
                        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
                    )
                ])
                pie_fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title=dict(
                        text="Bölge Dağılımı (%)" if selected_language == "Türkçe" else "Region Distribution (%)",
                        font=dict(color='white')
                    )
                )
                st.plotly_chart(pie_fig, use_container_width=True)
            
            with col2:
                st.subheader(texts['analysis_title'])
                st.markdown(texts['distribution_text'])

    except Exception as e:
        error_msg = "An error occurred" if selected_language == "English" else "Hata oluştu"
        retry_msg = "Please try another image or refresh the page" if selected_language == "English" else "Lütfen farklı bir görüntü deneyin veya sayfayı yenileyin"
        st.error(f"{error_msg}: {str(e)}")
        st.error(retry_msg)
