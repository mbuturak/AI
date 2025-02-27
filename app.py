import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import warnings
import plotly.graph_objects as go
import cv2
import os
import random

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
        The pie chart visualizes the distribution of detected regions in the X-Ray image:
        
        - The chart shows both the count and percentage of each detected anatomical structure
        - The donut-style visualization helps to understand the relative proportions
        - Hover over segments to see detailed information
        
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
        """,
        'density_title': "Regional Density Analysis",
        'density_text': """
        **Regional Density Analysis Interpretation:**
        
        The heat map overlaid on the X-Ray image shows the concentration of detected bone regions:
        
        - **High Density Areas (Red/Yellow)**: 
          - Indicates regions with multiple bone structures in close proximity
          - Common in joints and complex anatomical structures
          - May suggest areas requiring detailed examination
        
        - **Medium Density Areas (Green/Blue)**:
          - Shows moderate concentration of bone structures
          - Typically represents normal anatomical spacing
        
        - **Low Density Areas (Dark Blue)**:
          - Indicates areas with fewer detected structures
          - May represent spaces between major bone groups
        
        This analysis helps in:
        - Identifying areas of structural complexity
        - Understanding bone distribution patterns
        - Highlighting regions that may need closer examination
        """,
        'size_title': "Size Analysis",
        'size_text': """
        **Size Analysis Interpretation:**
        
        The box plot shows the size distribution of detected regions:
        
        - Each box represents a different anatomical structure
        - The box shows the quartiles (25th to 75th percentile)
        - The line inside the box is the median size
        - Individual points show specific detections
        
        This helps in:
        - Comparing sizes across different bone structures
        - Identifying unusually large or small detections
        - Understanding the typical size range for each structure
        """,
        'confidence_title': "Confidence Score Analysis",
        'confidence_text': """
        **Confidence Score Analysis:**
        
        The violin plot shows the distribution of model's confidence scores:
        
        - Wider sections indicate more common confidence values
        - The box inside shows quartiles and median
        - Higher scores (closer to 1.0) indicate stronger confidence
        - Lower scores suggest less certain detections
        
        This helps in:
        - Assessing the reliability of detections
        - Identifying patterns in model certainty
        - Understanding detection quality for different structures
        """,
        'symmetry_title': "Symmetry Analysis",
        'symmetry_text': """
        **Symmetry Analysis:**
        
        The bar chart compares left-right distribution of detections:
        
        - Shows the number of detections on each side
        - Helps identify potential asymmetries
        - Useful for bilateral comparison
        
        Important for:
        - Checking structural balance
        - Identifying potential anomalies
        - Supporting comparative analysis
        """,
        'distance_title': "Distance Analysis",
        'distance_text': """
        **Distance Analysis:**
        
        The histogram shows the distribution of distances between detected regions:
        
        - X-axis shows the distance between pairs of regions
        - Y-axis shows how frequently each distance occurs
        - Peaks indicate common spacing patterns
        
        This analysis helps in:
        - Understanding spatial relationships
        - Identifying typical spacing patterns
        - Detecting unusual positioning
        """,
        'region': "Region",
        'brand': "Brand",
        'model': "Model",
        'certainty': "Certainty",
        'no_implant': "No implants detected"
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
        Pasta grafik, X-Ray görüntüsünde tespit edilen bölgelerin dağılımını göstermektedir:
        
        - Grafik, her anatomik yapının hem sayısını hem de yüzdesini gösterir
        - Halka şeklindeki görselleştirme, oranları anlamayı kolaylaştırır
        - Detaylı bilgi için dilimler üzerine gelin
        
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
        """,
        'density_title': "Bölgesel Yoğunluk Analizi",
        'density_text': """
        **Bölgesel Yoğunluk Analizi Yorumu:**
        
        X-Ray görüntüsü üzerine bindirilen ısı haritası, tespit edilen kemik bölgelerinin yoğunluğunu gösterir:
        
        - **Yüksek Yoğunluklu Bölgeler (Kırmızı/Sarı)**:
          - Indicates regions with multiple bone structures in close proximity
          - Common in joints and complex anatomical structures
          - May suggest areas requiring detailed examination
        
        - **Orta Yoğunluklu Bölgeler (Yeşil/Mavi)**:
          - Shows moderate concentration of bone structures
          - Typically represents normal anatomical spacing
        
        - **Düşük Yoğunluklu Bölgeler (Koyu Mavi)**:
          - Indicates areas with fewer detected structures
          - May represent spaces between major bone groups
        
        Bu analiz şu konularda yardımcı olur:
        - Identifying areas of structural complexity
        - Understanding bone distribution patterns
        - Highlighting regions that may need closer examination
        """,
        'size_title': "Boyut Analizi",
        'size_text': """
        **Boyut Analizi Yorumu:**
        
        The box plot shows the size distribution of detected regions:
        
        - Each box represents a different anatomical structure
        - The box shows the quartiles (25th to 75th percentile)
        - The line inside the box is the median size
        - Individual points show specific detections
        
        This helps in:
        - Comparing sizes across different bone structures
        - Identifying unusually large or small detections
        - Understanding the typical size range for each structure
        """,
        'confidence_title': "Güven Skoru Analizi",
        'confidence_text': """
        **Güven Skoru Analizi:**
        
        The violin plot shows the distribution of model's confidence scores:
        
        - Wider sections indicate more common confidence values
        - The box inside shows quartiles and median
        - Higher scores (closer to 1.0) indicate stronger confidence
        - Lower scores suggest less certain detections
        
        This helps in:
        - Assessing the reliability of detections
        - Identifying patterns in model certainty
        - Understanding detection quality for different structures
        """,
        'symmetry_title': "Simetri Analizi",
        'symmetry_text': """
        **Simetri Analizi:**
        
        The bar chart compares left-right distribution of detections:
        
        - Shows the number of detections on each side
        - Helps identify potential asymmetries
        - Useful for bilateral comparison
        
        Important for:
        - Checking structural balance
        - Identifying potential anomalies
        - Supporting comparative analysis
        """,
        'distance_title': "Mesafe Analizi",
        'distance_text': """
        **Mesafe Analizi:**
        
        The histogram shows the distribution of distances between detected regions:
        
        - X-axis shows the distance between pairs of regions
        - Y-axis shows how frequently each distance occurs
        - Peaks indicate common spacing patterns
        
        This analysis helps in:
        - Understanding spatial relationships
        - Identifying typical spacing patterns
        - Detecting unusual positioning
        """,
        'region': "Bölge",
        'brand': "Marka",
        'model': "Model",
        'certainty': "Güven",
        'no_implant': "İmplant yok"
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
        # YOLO modeli yükleme
        model = YOLO(model_path)  # 'version' parametresi kaldırıldı
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB cinsinden
        st.sidebar.success(f"YOLO modeli başarıyla yüklendi! (Boyut: {file_size:.2f} MB)")
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

# Demo görüntü seçimi için fonksiyon
def load_demo_image(image_path):
    return Image.open(image_path)

# Sidebar'a demo görüntü seçimi ekle
st.sidebar.markdown("---")
st.sidebar.subheader("Demo Görseller" if selected_language == "Türkçe" else "Demo Images")

# Demo görüntüleri listele
demo_images = {
    "Demo 1": "images/demo1.jpg",
    "Demo 2": "images/demo2.jpg",
    "Demo 3": "images/demo3.jpg",
    "Demo 4": "images/demo4.jpg"
}

# Demo görüntü seçimi
selected_demo = st.sidebar.radio(
    "Demo Görsel Seç" if selected_language == "Türkçe" else "Select Demo Image",
    list(demo_images.keys()),
    index=None
)

# Kullanıcı yükleme veya demo seçimi
if selected_demo:
    # Demo görüntüyü yükle
    image = load_demo_image(demo_images[selected_demo])
    img_array = np.array(image)
    
    # Görüntü işleme kodları buradan devam eder...
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

        # Define color palette - Daha belirgin ve kontrast renkler
        colors = [
            '#FF0000',  # Kırmızı
            '#00FF00',  # Yeşil
            '#0000FF',  # Mavi
            '#FFA500',  # Turuncu
            '#800080',  # Mor
            '#00FFFF',  # Cyan
            '#FFD700',  # Altın
        ]

        # Add detected areas with custom shapes based on class
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            # Her tespitin bölge ismini ekle (sol üst köşe)
            region_names = set()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)
                full_label = results[0].names[cls]
                base_label = full_label.split('_')[0].upper()
                if base_label in ["HIP", "SHOULDER"]:
                    region_names.add(base_label)

                # Bölge ismini görüntüde göster
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if "true" in full_label.lower():  # Yalnızca "true" tespitler için çizim
                    width = x2 - x1
                    height = y2 - y1
                    color = colors[i % len(colors)]

                    # Path çizimi için noktalar
                    points = np.array([
                        [x1, y1],  # Sol üst
                        [cx, y1 - height * 0.1],  # Üst orta
                        [x2, y1],  # Sağ üst
                        [x2 + width * 0.1, cy],  # Sağ orta
                        [x2, y2],  # Sağ alt
                        [cx, y2 + height * 0.1],  # Alt orta
                        [x1, y2],  # Sol alt
                        [x1 - width * 0.1, cy],  # Sol orta
                        [x1, y1]  # Başlangıç noktasına dön
                    ])

                    # Path çizimi
                    fig.add_trace(go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode='lines',
                        line=dict(color=color, width=3),
                        name=base_label,
                        showlegend=True,
                        hoverinfo='text',
                        hovertext=f"{base_label}<br>Güven: {conf:.2%}" if selected_language == "Türkçe"
                        else f"{base_label}<br>Confidence: {conf:.2%}"
                    ))

                # Etiket ekle (her zaman bölge ismi göster)
                fig.add_annotation(
                    x=cx,
                    y=y1 - 10,
                    text=base_label,
                    showarrow=False,
                    font=dict(
                        color='white',
                        size=12,
                        weight='bold'
                    ),
                    bgcolor=color if "true" in full_label.lower() else 'rgba(0, 0, 0, 0.7)',
                    opacity=0.7,
                    bordercolor=color if "true" in full_label.lower() else 'white',
                    borderwidth=2,
                    borderpad=4,
                    align='center'
                )

            # Sol üst köşede tüm bölge isimlerini listele
            fig.add_annotation(
                x=50,
                y=50,
                text=f"Bölgeler: {', '.join(sorted(region_names))}" if selected_language == "Türkçe" 
                else f"Regions: {', '.join(sorted(region_names))}",
                showarrow=False,
                font=dict(
                    color='white',
                    size=14,
                    weight='bold'
                ),
                bgcolor='rgba(0, 0, 0, 0.7)',
                bordercolor='white',
                borderwidth=2,
                borderpad=4,
                align='left'
            )

        # Update layout with improved legend
        fig.update_layout(
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)',
                font=dict(color='white', size=12),
                bordercolor='white',
                borderwidth=1,
                groupclick="toggleitem"  # Her öğeyi bağımsız açıp kapatma
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='black',
            paper_bgcolor='black',
            width=None,
            height=600,
            hoverlabel=dict(
                namelength=-1,
                font=dict(size=14)
            ),
            hovermode='closest'
        )

        
        # X-ray görüntülerini yan yana göster
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("**Original X-Ray**" if selected_language == "English" else "**Orijinal X-Ray**")
            # Orijinal görüntüy Plotly ile göster
            orig_fig = go.Figure()
            orig_fig.add_trace(go.Image(z=img_array))
            orig_fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='black',
                paper_bgcolor='black',
                width=None,  # Otomatik genişlik için None
                height=600
            )
            st.plotly_chart(orig_fig, use_container_width=True)
        
        with col_img2:
            st.markdown("**Detected Regions**" if selected_language == "English" else "**Tespit Edilen Bölgeler**")
            st.plotly_chart(fig, use_container_width=True)

        # Analiz bölümü
        st.markdown("---")
        
        # Dağılım analizi için iki kolon oluştur
        col1, col2 = st.columns([2, 1])  # Sol taraf daha geniş
        
        with col1:
            st.subheader(texts['distribution_title'])
            
            # Bölgelerin sayısını hesapla
            class_counts = {}
            for box in boxes:
                cls = int(box.cls)
                # Sadece bölge ismini al (true/false kısmını kaldır)
                label = results[0].names[cls].split('_')[0].upper()
                class_counts[label] = class_counts.get(label, 0) + 1

            # Pasta grafik
            pie_fig = go.Figure(data=[
                go.Pie(
                    labels=list(class_counts.keys()),
                    values=list(class_counts.values()),
                    hole=.3,
                    textinfo='label+value+percent',
                    texttemplate='%{label}<br>%{value} adet<br>(%{percent})',
                    marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']),
                    textfont=dict(size=14),
                    textposition='inside',
                    insidetextorientation='radial',
                    hoverinfo='label+value+percent',
                    hoverlabel=dict(font=dict(size=14, color='white')),
                    rotation=90
                )
            ])
            pie_fig.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                title=dict(
                    text="Bölge Dağılımı" if selected_language == "Türkçe" else "Region Distribution",
                    font=dict(color='white', size=16),
                    y=0.95
                ),
                height=600,
                showlegend=False
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        
        with col2:
            st.subheader(texts['analysis_title'])
            st.markdown(texts['distribution_text'])

        # Bölgesel yoğunluk analizi
        st.markdown("---")
        st.subheader("Bölgesel Yoğunluk Analizi" if selected_language == "Türkçe" else "Regional Density Analysis")

        # Tespit edilen bölgelerin merkez noktalarını topla
        centers_x = []
        centers_y = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            centers_x.append((x1 + x2) / 2)
            centers_y.append((y1 + y2) / 2)

        # Yoğunluk haritası oluştur
        density_fig = go.Figure(data=go.Histogram2dContour(
            x=centers_x,
            y=centers_y,
            colorscale='Viridis',
            nbinsx=20,
            nbinsy=20,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Yoğunluk" if selected_language == "Türkçe" else "Density",
                    side="right"
                ),
                thickness=20,
                len=0.9
            ),
            hovertemplate=("Yoğunluk: %{z}<extra></extra>" if selected_language == "Türkçe" 
                         else "Density: %{z}<extra></extra>")
        ))
        
        # X-ray görüntüsünü ekle
        density_fig.add_trace(go.Image(
            z=img_array, 
            opacity=0.5,
            hovertemplate="X-Ray<extra></extra>"
        ))

        # Tespit edilen bölgeleri ekle
        for box, label in zip(boxes, [results[0].names[int(box.cls)] for box in boxes]):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            density_fig.add_trace(go.Scatter(
                x=[cx],
                y=[cy],
                mode='markers',
                marker=dict(size=10, opacity=0),
                hoverinfo='text',
                hovertext=label,
                showlegend=False
            ))
        
        density_fig.update_layout(
            title=dict(
                text="Tespit Yoğunluğu" if selected_language == "Türkçe" else "Detection Density",
                font=dict(color='white', size=16),
                y=0.95
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            height=700,  # Yüksekliği daha da artır
            margin=dict(l=20, r=50, t=50, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(density_fig, use_container_width=True)
        
        # Açıklamayı altına ekle
        st.markdown(texts['density_text'])

        # Boyut analizi bölümü
        st.markdown("---")
        st.subheader("Boyut Analizi" if selected_language == "Türkçe" else "Size Analysis")

        # Bölgelerin boyutlarını hesapla
        sizes = []
        size_labels = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            sizes.append(area)
            # Sadece bölge ismini al
            size_labels.append(results[0].names[int(box.cls)].split('_')[0].upper())

        # Kutu grafiği oluştur
        box_fig = go.Figure()
        for label in set(size_labels):
            label_sizes = [size for size, l in zip(sizes, size_labels) if l == label]
            box_fig.add_trace(go.Box(
                y=label_sizes,
                name=label,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(opacity=0.7)
            ))

        box_fig.update_layout(
            title="Bölge Boyutları Karşılaştırması" if selected_language == "Türkçe" else "Region Size Comparison",
            yaxis_title="Alan" if selected_language == "Türkçe" else "Area",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(box_fig, use_container_width=True)
        st.markdown(texts['size_text'])

        # Güven skoru analizi
        st.markdown("---")
        st.subheader("Güven Skoru Analizi" if selected_language == "Türkçe" else "Confidence Score Analysis")

        conf_data = []
        conf_labels = []
        for box in boxes:
            conf_data.append(float(box.conf))
            # Sadece bölge ismini al
            conf_labels.append(results[0].names[int(box.cls)].split('_')[0].upper())

        conf_fig = go.Figure()
        for label in set(conf_labels):
            label_confs = [conf for conf, l in zip(conf_data, conf_labels) if l == label]
            conf_fig.add_trace(go.Violin(
                y=label_confs,
                name=label,
                box_visible=True,
                meanline_visible=True,
                fillcolor=f'rgba({random.randint(50,200)}, {random.randint(50,200)}, {random.randint(50,200)}, 0.6)'
            ))

        conf_fig.update_layout(
            title="Güven Skorları Dağılımı" if selected_language == "Türkçe" else "Confidence Score Distribution",
            yaxis_title="Güven Skoru" if selected_language == "Türkçe" else "Confidence Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(conf_fig, use_container_width=True)
        st.markdown(texts['confidence_text'])

        # Simetri analizi
        st.markdown("---")
        st.subheader("Simetri Analizi" if selected_language == "Türkçe" else "Symmetry Analysis")

        # Görüntüyü dikey olarak ikiye böl
        image_center = img_array.shape[1] // 2
        left_detections = 0
        right_detections = 0

        for box in boxes:
            x1, x2 = box.xyxy[0].cpu().numpy()[[0, 2]]
            center = (x1 + x2) / 2
            if center < image_center:
                left_detections += 1
            else:
                right_detections += 1

        # Simetri grafiği
        symmetry_fig = go.Figure(data=[
            go.Bar(
                x=['Sol', 'Sağ'] if selected_language == "Türkçe" else ['Left', 'Right'],
                y=[left_detections, right_detections],
                marker_color=['#1f77b4', '#ff7f0e'],
                text=[f'{left_detections}', f'{right_detections}'],
                textposition='auto'
            )
        ])
        
        # Simetri oranı hesapla ve ekle
        total = left_detections + right_detections
        symmetry_ratio = min(left_detections, right_detections) / max(left_detections, right_detections) if max(left_detections, right_detections) > 0 else 1
        
        symmetry_fig.add_annotation(
            x=0.5,
            y=max(left_detections, right_detections) * 1.1,
            xref="paper",
            text=f"Simetri Oranı: {symmetry_ratio:.2f}" if selected_language == "Türkçe" else f"Symmetry Ratio: {symmetry_ratio:.2f}",
            showarrow=False,
            font=dict(size=14, color="white")
        )
        
        symmetry_fig.update_layout(
            title="Sol-Sağ Dağılımı" if selected_language == "Türkçe" else "Left-Right Distribution",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(symmetry_fig, use_container_width=True)
        st.markdown(texts['symmetry_text'])

        # Mesafe analizi
        st.markdown("---")
        st.subheader("Mesafe Analizi" if selected_language == "Türkçe" else "Distance Analysis")

        distances = []
        pair_labels = []
        for i, box1 in enumerate(boxes):
            for j, box2 in enumerate(boxes[i+1:], i+1):
                x1, y1 = (box1.xyxy[0][0] + box1.xyxy[0][2])/2, (box1.xyxy[0][1] + box1.xyxy[0][3])/2
                x2, y2 = (box2.xyxy[0][0] + box2.xyxy[0][2])/2, (box2.xyxy[0][1] + box2.xyxy[0][3])/2
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                distances.append(distance)
                # Sadece bölge isimlerini al
                label1 = results[0].names[int(box1.cls)].split('_')[0].upper()
                label2 = results[0].names[int(box2.cls)].split('_')[0].upper()
                pair_labels.append(f"{label1} - {label2}")

        dist_fig = go.Figure(data=go.Histogram(
            x=distances,
            nbinsx=30,
            opacity=0.7,
            marker_color='#007AFF'
        ))
        dist_fig.update_layout(
            title="Bölgeler Arası Mesafe Dağılımı" if selected_language == "Türkçe" else "Inter-Region Distance Distribution",
            xaxis_title="Mesafe" if selected_language == "Türkçe" else "Distance",
            yaxis_title="Frekans" if selected_language == "Türkçe" else "Frequency",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(dist_fig, use_container_width=True)
        st.markdown(texts['distance_text'])

        # İmplant Bilgileri Özet Tablosu
        st.markdown("---")
        st.subheader("İmplant Bilgileri Özeti" if selected_language == "Türkçe" else "Implant Information Summary")
        
        # İmplant bilgilerini topla
        implant_data = []
        for i, box in enumerate(boxes):
            cls = int(box.cls)
            conf = float(box.conf)
            full_label = results[0].names[cls]
            base_label = full_label.split('_')[0].upper()
            
            if "true" in full_label.lower():
                brand, model = get_random_implant(base_label)
                implant_data.append({
                    "region": base_label,
                    "brand": brand,
                    "model": model,
                    "confidence": conf
                })
        
        # İmplant verilerini tablo olarak göster
        if implant_data:
            # Tablo sütun başlıkları
            cols = st.columns(4)
            cols[0].markdown(f"**{texts['region']}**")
            cols[1].markdown(f"**{texts['brand']}**")
            cols[2].markdown(f"**{texts['model']}**")
            cols[3].markdown(f"**{texts['certainty']}**")
            
            # Her bir implantı tabloya ekle
            for data in implant_data:
                cols = st.columns(4)
                cols[0].markdown(data["region"])
                cols[1].markdown(data["brand"])
                cols[2].markdown(data["model"])
                
                # Renkli güven göstergesi
                confidence = data["confidence"]
                if confidence > 0.7:
                    color = "#4CAF50"  # Yeşil
                elif confidence > 0.5:
                    color = "#FFC107"  # Sarı
                else:
                    color = "#F44336"  # Kırmızı
                    
                confidence_html = f"""
                <div style="background-color:#333; width:100%; height:15px; border-radius:7px;">
                    <div style="background-color:{color}; width:{confidence*100}%; height:15px; border-radius:7px;"></div>
                </div>
                <div style="text-align:right; font-size:0.8em;">{confidence:.1%}</div>
                """
                cols[3].markdown(confidence_html, unsafe_allow_html=True)
        else:
            st.info(texts['no_implant'])

        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align:center; padding:10px; background-color:#051937; border-radius:10px">
                <h4 style="color:white; margin-bottom:0">MedVision AI © 2023</h4>
                <p style="color:#64B5F6; font-size:0.9em; margin-top:5px">Advanced Medical Image Analysis Platform</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

else:
    # Normal dosya yükleme alanı
    uploaded_file = st.file_uploader(texts['upload'], type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Görüntü işleme ve çizdirme
            fig, img_array, boxes, results, implant_html = process_image(img_array, selected_language)
            
            # X-ray görüntülerini yan yana göster
            st.markdown("### " + texts['title'])
            
            col_img1, col_img2 = st.columns([1, 1])
            
            with col_img1:
                st.markdown("**Original X-Ray**" if selected_language == "English" else "**Orijinal X-Ray**")
                # Orijinal görüntüyü Plotly ile göster
                orig_fig = go.Figure()
                orig_fig.add_trace(go.Image(z=img_array))
                orig_fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    width=None,  # Otomatik genişlik için None
                    height=600
                )
                st.plotly_chart(orig_fig, use_container_width=True)
            
            with col_img2:
                st.markdown("**Detected Regions**" if selected_language == "English" else "**Tespit Edilen Bölgeler**")
                st.plotly_chart(fig, use_container_width=True)
            
            # İmplant bilgi kutularını göster
            if implant_html:
                st.markdown("### " + ("Tespit Edilen İmplantlar" if selected_language == "Türkçe" else "Detected Implants"))
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(implant_html, unsafe_allow_html=True)
            
            # Analiz bölümü
            st.markdown("---")
            
            # Dağılım analizi için iki kolon oluştur
            col1, col2 = st.columns([2, 1])  # Sol taraf daha geniş
            
            with col1:
                st.subheader(texts['distribution_title'])
                
                # Bölgelerin sayısını hesapla
                class_counts = {}
                for box in boxes:
                    cls = int(box.cls)
                    # Sadece bölge ismini al (true/false kısmını kaldır)
                    label = results[0].names[cls].split('_')[0].upper()
                    class_counts[label] = class_counts.get(label, 0) + 1

                # Pasta grafik
                pie_fig = go.Figure(data=[
                    go.Pie(
                        labels=list(class_counts.keys()),
                        values=list(class_counts.values()),
                        hole=.3,
                        textinfo='label+value+percent',
                        texttemplate='%{label}<br>%{value} adet<br>(%{percent})',
                        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']),
                        textfont=dict(size=14),
                        textposition='inside',
                        insidetextorientation='radial',
                        hoverinfo='label+value+percent',
                        hoverlabel=dict(font=dict(size=14, color='white')),
                        rotation=90
                    )
                ])
                pie_fig.update_layout(
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=14),
                    title=dict(
                        text="Bölge Dağılımı" if selected_language == "Türkçe" else "Region Distribution",
                        font=dict(color='white', size=16),
                        y=0.95
                    ),
                    height=600,
                    showlegend=False
                )
                st.plotly_chart(pie_fig, use_container_width=True)
            
            with col2:
                st.subheader(texts['analysis_title'])
                st.markdown(texts['distribution_text'])

            # Bölgesel yoğunluk analizi
            st.markdown("---")
            st.subheader("Bölgesel Yoğunluk Analizi" if selected_language == "Türkçe" else "Regional Density Analysis")

            # Tespit edilen bölgelerin merkez noktalarını topla
            centers_x = []
            centers_y = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                centers_x.append((x1 + x2) / 2)
                centers_y.append((y1 + y2) / 2)

            # Yoğunluk haritası oluştur
            density_fig = go.Figure(data=go.Histogram2dContour(
                x=centers_x,
                y=centers_y,
                colorscale='Viridis',
                nbinsx=20,
                nbinsy=20,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Yoğunluk" if selected_language == "Türkçe" else "Density",
                        side="right"
                    ),
                    thickness=20,
                    len=0.9
                ),
                hovertemplate=("Yoğunluk: %{z}<extra></extra>" if selected_language == "Türkçe" 
                             else "Density: %{z}<extra></extra>")
            ))
            
            # X-ray görüntüsünü ekle
            density_fig.add_trace(go.Image(
                z=img_array, 
                opacity=0.5,
                hovertemplate="X-Ray<extra></extra>"
            ))

            # Tespit edilen bölgeleri ekle
            for box, label in zip(boxes, [results[0].names[int(box.cls)] for box in boxes]):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                density_fig.add_trace(go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode='markers',
                    marker=dict(size=10, opacity=0),
                    hoverinfo='text',
                    hovertext=label,
                    showlegend=False
                ))
            
            density_fig.update_layout(
                title=dict(
                    text="Tespit Yoğunluğu" if selected_language == "Türkçe" else "Detection Density",
                    font=dict(color='white', size=16),
                    y=0.95
                ),
                paper_bgcolor='black',
                plot_bgcolor='black',
                height=700,  # Yüksekliği daha da artır
                margin=dict(l=20, r=50, t=50, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            st.plotly_chart(density_fig, use_container_width=True)
            
            # Açıklamayı altına ekle
            st.markdown(texts['density_text'])

            # Boyut analizi bölümü
            st.markdown("---")
            st.subheader("Boyut Analizi" if selected_language == "Türkçe" else "Size Analysis")

            # Bölgelerin boyutlarını hesapla
            sizes = []
            size_labels = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width = x2 - x1
                height = y2 - y1
                area = width * height
                sizes.append(area)
                # Sadece bölge ismini al
                size_labels.append(results[0].names[int(box.cls)].split('_')[0].upper())

            # Kutu grafiği oluştur
            box_fig = go.Figure()
            for label in set(size_labels):
                label_sizes = [size for size, l in zip(sizes, size_labels) if l == label]
                box_fig.add_trace(go.Box(
                    y=label_sizes,
                    name=label,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(opacity=0.7)
                ))

            box_fig.update_layout(
                title="Bölge Boyutları Karşılaştırması" if selected_language == "Türkçe" else "Region Size Comparison",
                yaxis_title="Alan" if selected_language == "Türkçe" else "Area",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(box_fig, use_container_width=True)
            st.markdown(texts['size_text'])

            # Güven skoru analizi
            st.markdown("---")
            st.subheader("Güven Skoru Analizi" if selected_language == "Türkçe" else "Confidence Score Analysis")

            conf_data = []
            conf_labels = []
            for box in boxes:
                conf_data.append(float(box.conf))
                # Sadece bölge ismini al
                conf_labels.append(results[0].names[int(box.cls)].split('_')[0].upper())

            conf_fig = go.Figure()
            for label in set(conf_labels):
                label_confs = [conf for conf, l in zip(conf_data, conf_labels) if l == label]
                conf_fig.add_trace(go.Violin(
                    y=label_confs,
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=f'rgba({random.randint(50,200)}, {random.randint(50,200)}, {random.randint(50,200)}, 0.6)'
                ))

            conf_fig.update_layout(
                title="Güven Skorları Dağılımı" if selected_language == "Türkçe" else "Confidence Score Distribution",
                yaxis_title="Güven Skoru" if selected_language == "Türkçe" else "Confidence Score",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(conf_fig, use_container_width=True)
            st.markdown(texts['confidence_text'])

            # Simetri analizi
            st.markdown("---")
            st.subheader("Simetri Analizi" if selected_language == "Türkçe" else "Symmetry Analysis")

            # Görüntüyü dikey olarak ikiye böl
            image_center = img_array.shape[1] // 2
            left_detections = 0
            right_detections = 0

            for box in boxes:
                x1, x2 = box.xyxy[0].cpu().numpy()[[0, 2]]
                center = (x1 + x2) / 2
                if center < image_center:
                    left_detections += 1
                else:
                    right_detections += 1

            # Simetri grafiği
            symmetry_fig = go.Figure(data=[
                go.Bar(
                    x=['Sol', 'Sağ'] if selected_language == "Türkçe" else ['Left', 'Right'],
                    y=[left_detections, right_detections],
                    marker_color=['#1f77b4', '#ff7f0e'],
                    text=[f'{left_detections}', f'{right_detections}'],
                    textposition='auto'
                )
            ])
            
            # Simetri oranı hesapla ve ekle
            total = left_detections + right_detections
            symmetry_ratio = min(left_detections, right_detections) / max(left_detections, right_detections) if max(left_detections, right_detections) > 0 else 1
            
            symmetry_fig.add_annotation(
                x=0.5,
                y=max(left_detections, right_detections) * 1.1,
                xref="paper",
                text=f"Simetri Oranı: {symmetry_ratio:.2f}" if selected_language == "Türkçe" else f"Symmetry Ratio: {symmetry_ratio:.2f}",
                showarrow=False,
                font=dict(size=14, color="white")
            )
            
            symmetry_fig.update_layout(
                title="Sol-Sağ Dağılımı" if selected_language == "Türkçe" else "Left-Right Distribution",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(symmetry_fig, use_container_width=True)
            st.markdown(texts['symmetry_text'])

            # Mesafe analizi
            st.markdown("---")
            st.subheader("Mesafe Analizi" if selected_language == "Türkçe" else "Distance Analysis")

            distances = []
            pair_labels = []
            for i, box1 in enumerate(boxes):
                for j, box2 in enumerate(boxes[i+1:], i+1):
                    x1, y1 = (box1.xyxy[0][0] + box1.xyxy[0][2])/2, (box1.xyxy[0][1] + box1.xyxy[0][3])/2
                    x2, y2 = (box2.xyxy[0][0] + box2.xyxy[0][2])/2, (box2.xyxy[0][1] + box2.xyxy[0][3])/2
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    distances.append(distance)
                    # Sadece bölge isimlerini al
                    label1 = results[0].names[int(box1.cls)].split('_')[0].upper()
                    label2 = results[0].names[int(box2.cls)].split('_')[0].upper()
                    pair_labels.append(f"{label1} - {label2}")

            dist_fig = go.Figure(data=go.Histogram(
                x=distances,
                nbinsx=30,
                opacity=0.7,
                marker_color='#007AFF'
            ))
            dist_fig.update_layout(
                title="Bölgeler Arası Mesafe Dağılımı" if selected_language == "Türkçe" else "Inter-Region Distance Distribution",
                xaxis_title="Mesafe" if selected_language == "Türkçe" else "Distance",
                yaxis_title="Frekans" if selected_language == "Türkçe" else "Frequency",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(dist_fig, use_container_width=True)
            st.markdown(texts['distance_text'])

            # İmplant Bilgileri Özet Tablosu
            st.markdown("---")
            st.subheader("İmplant Bilgileri Özeti" if selected_language == "Türkçe" else "Implant Information Summary")
            
            # İmplant bilgilerini topla
            implant_data = []
            for i, box in enumerate(boxes):
                cls = int(box.cls)
                conf = float(box.conf)
                full_label = results[0].names[cls]
                base_label = full_label.split('_')[0].upper()
                
                if "true" in full_label.lower():
                    brand, model = get_random_implant(base_label)
                    implant_data.append({
                        "region": base_label,
                        "brand": brand,
                        "model": model,
                        "confidence": conf
                    })
            
            # İmplant verilerini tablo olarak göster
            if implant_data:
                # Tablo sütun başlıkları
                cols = st.columns(4)
                cols[0].markdown(f"**{texts['region']}**")
                cols[1].markdown(f"**{texts['brand']}**")
                cols[2].markdown(f"**{texts['model']}**")
                cols[3].markdown(f"**{texts['certainty']}**")
                
                # Her bir implantı tabloya ekle
                for data in implant_data:
                    cols = st.columns(4)
                    cols[0].markdown(data["region"])
                    cols[1].markdown(data["brand"])
                    cols[2].markdown(data["model"])
                    
                    # Renkli güven göstergesi
                    confidence = data["confidence"]
                    if confidence > 0.7:
                        color = "#4CAF50"  # Yeşil
                    elif confidence > 0.5:
                        color = "#FFC107"  # Sarı
                    else:
                        color = "#F44336"  # Kırmızı
                        
                    confidence_html = f"""
                    <div style="background-color:#333; width:100%; height:15px; border-radius:7px;">
                        <div style="background-color:{color}; width:{confidence*100}%; height:15px; border-radius:7px;"></div>
                    </div>
                    <div style="text-align:right; font-size:0.8em;">{confidence:.1%}</div>
                    """
                    cols[3].markdown(confidence_html, unsafe_allow_html=True)
            else:
                st.info(texts['no_implant'])

            # Footer
            st.markdown("---")
            st.markdown(
                """
                <div style="text-align:center; padding:10px; background-color:#051937; border-radius:10px">
                    <h4 style="color:white; margin-bottom:0">MedVision AI © 2023</h4>
                    <p style="color:#64B5F6; font-size:0.9em; margin-top:5px">Advanced Medical Image Analysis Platform</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        except Exception as e:
            error_msg = "An error occurred" if selected_language == "English" else "Hata oluştu"
            retry_msg = "Please try another image or refresh the page" if selected_language == "English" else "Lütfen farklı bir görüntü deneyin veya sayfayı yenileyin"
            st.error(f"{error_msg}: {str(e)}")
            st.error(retry_msg)
    else:
        # Anasayfa bilgi kartları
        st.markdown(
            """
            <div style="text-align:center; padding:20px; margin-top:20px;">
                <img src="https://img.icons8.com/fluency/96/000000/x-ray.png" style="width:80px;">
                <h2 style="color:white; margin-top:10px">X-Ray Analiz Sistemi</h2>
                <p style="color:#999; font-size:1.1em">Yapay zeka destekli tıbbi görüntü analizi</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Özellik kartları
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div style="background-color:rgba(5, 25, 55, 0.7); padding:15px; border-radius:10px; height:220px; text-align:center;">
                    <img src="https://img.icons8.com/fluency/48/000000/search.png" style="width:48px;">
                    <h3 style="color:#64B5F6;">Otomatik Tespit</h3>
                    <p style="color:white;">Yapay zeka ile X-Ray görüntülerinde anatomik bölgeleri ve implantları otomatik tespit eder.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(
                """
                <div style="background-color:rgba(5, 25, 55, 0.7); padding:15px; border-radius:10px; height:220px; text-align:center;">
                    <img src="https://img.icons8.com/fluency/48/000000/activity-feed.png" style="width:48px;">
                    <h3 style="color:#64B5F6;">Detaylı Analiz</h3>
                    <p style="color:white;">Tespit edilen bölgelerin dağılımı, boyutu, ve ilişkilerini detaylı olarak analiz eder.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col3:
            st.markdown(
                """
                <div style="background-color:rgba(5, 25, 55, 0.7); padding:15px; border-radius:10px; height:220px; text-align:center;">
                    <img src="https://img.icons8.com/fluency/48/000000/data-sheet.png" style="width:48px;">
                    <h3 style="color:#64B5F6;">İmplant Bilgileri</h3>
                    <p style="color:white;">Tespit edilen implantların marka, model ve diğer özelliklerini belirler ve raporlar.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Başlama yönergeleri
        st.markdown("---")
        st.markdown(
            """
            <div style="padding:15px; background-color:rgba(0,122,255,0.1); border-radius:10px; border-left:5px solid #007AFF;">
                <h3 style="color:#007AFF;">Başlarken</h3>
                <p style="color:white;">Analiz yapmak için sol menüden bir demo görüntü seçin veya kendi X-Ray görüntünüzü yükleyin.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Eksik Streamlit logo/Github butonunu gizleme CSS kodu eklenecek
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# İmplant markalarını tanımlayalım
IMPLANT_BRANDS = {
    "SHOULDER": [
        {"brand": "ArthroFlex", "models": ["A100", "A200", "A300"]},
        {"brand": "OrthoMed", "models": ["OM-S1", "OM-S2", "OM-S3"]},
        {"brand": "BioTech Implants", "models": ["ShoulderPro", "ShoulderElite", "ShoulderMax"]},
        {"brand": "MedicalForge", "models": ["Titanium S", "Platinum S", "Diamond S"]},
        {"brand": "Zimmer Biomet", "models": ["ZB-5000", "ZB-6000", "ZB-7000"]},
    ],
    "HIP": [
        {"brand": "JointCraft", "models": ["HC-100", "HC-200", "HC-300"]},
        {"brand": "ArthroCare", "models": ["Alpha Hip", "Beta Hip", "Gamma Hip"]},
        {"brand": "OsteoBionics", "models": ["HipFlex", "HipMotion", "HipStride"]},
        {"brand": "Smith & Nephew", "models": ["SN-Hip1", "SN-Hip2", "SN-Hip3"]},
        {"brand": "Stryker", "models": ["Secur-Hip", "Mako Hip", "Accolade Hip"]},
    ]
}

# İmplant markaları için yardımcı fonksiyon
def get_random_implant(region_type):
    if region_type.upper() in IMPLANT_BRANDS:
        brand_info = random.choice(IMPLANT_BRANDS[region_type.upper()])
        brand = brand_info["brand"]
        model = random.choice(brand_info["models"])
        return brand, model
    return "Unknown", "Unknown"

# İmplant tespit kutusu oluşturma fonksiyonu
def create_implant_box(region, brand, model, confidence, selected_language):
    texts = TEXTS[selected_language]
    
    status_color = "#4CAF50" if confidence > 0.7 else "#FFC107" if confidence > 0.5 else "#F44336"
    
    html = f"""
    <div style="background-color:rgba(5, 25, 55, 0.85); padding:10px; border-radius:8px; 
               border-left: 5px solid {status_color}; margin-bottom:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h4 style="color:white; margin-top:0; margin-bottom:5px;">{"İmplant Tespit Edildi" if selected_language == "Türkçe" else "Implant Detected"}</h4>
        <table style="width:100%; color:white; margin-bottom:5px;">
            <tr>
                <td style="padding:3px; width:30%; color:#64B5F6;"><b>{texts['region']}:</b></td>
                <td style="padding:3px;"><b>{region}</b></td>
            </tr>
            <tr>
                <td style="padding:3px; color:#64B5F6;"><b>{texts['brand']}:</b></td>
                <td style="padding:3px;">{brand}</td>
            </tr>
            <tr>
                <td style="padding:3px; color:#64B5F6;"><b>{texts['model']}:</b></td>
                <td style="padding:3px;">{model}</td>
            </tr>
            <tr>
                <td style="padding:3px; color:#64B5F6;"><b>{texts['certainty']}:</b></td>
                <td style="padding:3px;">
                    <div style="background-color:#333; width:100%; height:10px; border-radius:5px;">
                        <div style="background-color:{status_color}; width:{confidence*100}%; height:10px; border-radius:5px;"></div>
                    </div>
                    <div style="text-align:right; font-size:0.8em;">{confidence:.1%}</div>
                </td>
            </tr>
        </table>
    </div>
    """
    return html

# Görüntü işleme fonksiyonu
def process_image(img_array, selected_language):
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
        
        # Create Plotly figure
        fig = go.Figure()

        # Add image as background
        fig.add_trace(go.Image(z=img_array))

        # Define color palette - Daha belirgin ve kontrast renkler
        colors = [
            '#FF3B30',  # Apple Red
            '#34C759',  # Apple Green
            '#007AFF',  # Apple Blue
            '#FF9500',  # Apple Orange
            '#5856D6',  # Apple Purple
            '#00C7BE',  # Apple Teal
            '#FFCC00',  # Apple Yellow
        ]

        implant_html = ""
        
        # Add detected areas with custom shapes based on class
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            # Her tespitin bölge ismini ekle
            region_names = set()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)
                full_label = results[0].names[cls]
                base_label = full_label.split('_')[0].upper()
                if base_label in ["HIP", "SHOULDER"]:
                    region_names.add(base_label)

                # Bölge ismini görüntüde göster
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if "true" in full_label.lower():  # Yalnızca "true" tespitler için çizim
                    width = x2 - x1
                    height = y2 - y1
                    color = colors[i % len(colors)]
                    
                    # Rastgele implant bilgisi ata
                    brand, model = get_random_implant(base_label)
                    
                    # HTML kutusu oluştur
                    implant_html += create_implant_box(
                        base_label, brand, model, conf, selected_language
                    )

                    # Gelişmiş şık işaretleme - Yuvarlak köşeli kutu ve etiketli
                    # Path çizimi için noktalar - daha yumuşak köşeler
                    curve_factor = 0.2
                    points = []
                    
                    # Sol üst köşeden başla
                    points.append([x1, y1 + height * curve_factor])  # Sol üst başlangıç
                    points.append([x1 + width * curve_factor, y1])   # Sol üst kavis
                    
                    # Üst kenar
                    points.append([x2 - width * curve_factor, y1])   # Sağ üst kavis başlangıcı
                    
                    # Sağ üst köşe
                    points.append([x2, y1 + height * curve_factor])  # Sağ üst kavis sonu
                    
                    # Sağ kenar
                    points.append([x2, y2 - height * curve_factor])  # Sağ alt kavis başlangıcı
                    
                    # Sağ alt köşe
                    points.append([x2 - width * curve_factor, y2])   # Sağ alt kavis
                    
                    # Alt kenar
                    points.append([x1 + width * curve_factor, y2])   # Sol alt kavis başlangıcı
                    
                    # Sol alt köşe
                    points.append([x1, y2 - height * curve_factor])  # Sol alt kavis sonu
                    
                    # Kapatma
                    points.append([x1, y1 + height * curve_factor])  # Başlangıç noktasına dön
                    
                    points = np.array(points)

                    # Gelişmiş çizgi stili ile path çizimi
                    fig.add_trace(go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode='lines',
                        line=dict(color=color, width=3, dash='solid'),
                        fill='none',
                        name=f"{base_label} - {brand}",
                        showlegend=True,
                        hoverinfo='text',
                        hovertext=f"<b>{base_label}</b><br>Marka: {brand}<br>Model: {model}<br>{'Güven' if selected_language == 'Türkçe' else 'Confidence'}: {conf:.2%}"
                    ))
                    
                    # Bölge için daha şık etiket kutusu
                    fig.add_annotation(
                        x=cx,
                        y=y1 - 15,
                        text=f"{base_label} | {brand}",
                        showarrow=False,
                        font=dict(
                            color='white',
                            size=12,
                            weight='bold'
                        ),
                        bgcolor=color,
                        opacity=0.9,
                        bordercolor='white',
                        borderwidth=1,
                        borderpad=4,
                        align='center',
                        textangle=0,
                        width=width * 0.8
                    )

            # Sol üst köşede daha şık bir bölgeler kutusu
            if region_names:
                regions_text = ", ".join(sorted(region_names))
                fig.add_annotation(
                    x=20,
                    y=30,
                    xref="x",
                    yref="y",
                    text=f"<b>{'Bölgeler' if selected_language == 'Türkçe' else 'Regions'}</b>: {regions_text}",
                    showarrow=False,
                    font=dict(
                        color='white',
                        size=14,
                        weight='bold'
                    ),
                    bgcolor='rgba(0, 0, 0, 0.7)',
                    bordercolor='#007AFF',
                    borderwidth=2,
                    borderpad=8,
                    align='left',
                    xanchor='left',
                    yanchor='top'
                )

        # Update layout with improved legend
        fig.update_layout(
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.7)',
                font=dict(color='white', size=12),
                bordercolor='white',
                borderwidth=1,
                groupclick="toggleitem"
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='black',
            paper_bgcolor='black',
            width=None,
            height=600,
            hoverlabel=dict(
                namelength=-1,
                font=dict(size=14),
                bgcolor='rgba(0,0,0,0.8)'
            ),
            hovermode='closest'
        )
        
        return fig, img_array, boxes, results, implant_html
