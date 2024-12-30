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
                
                # Önce bölge ismini ekle (sol üst köşe)
                if len(boxes) > 0:
                    # İlk tespitin sınıfını al ve sadece bölge ismini kullan
                    first_cls = int(boxes[0].cls)
                    region_name = results[0].names[first_cls].split('_')[0].upper()  # true/false kısmını kaldır
                    
                    # Bölge ismini sol üst köşeye ekle
                    fig.add_annotation(
                        x=50,
                        y=50,
                        text=f"Bölge: {region_name}" if selected_language == "Türkçe" else f"Region: {region_name}",
                        showarrow=False,
                        font=dict(
                            color='white',
                            size=16,
                            weight='bold'
                        ),
                        bgcolor='rgba(0,0,0,0.7)',
                        bordercolor='white',
                        borderwidth=2,
                        borderpad=4,
                        align='left',
                        xanchor='left',
                        yanchor='top'
                    )
                
                # Sonra normal tespitleri işle
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    cls = int(box.cls)
                    full_label = results[0].names[cls]
                    # Sadece bölge ismini al (true/false kısmını kaldır)
                    base_label = full_label.split('_')[0].upper()
                    is_true = "true" in full_label.lower()
                    
                    # Merkez noktaları
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Renk seç
                    color = colors[i % len(colors)]

                    # Sadece true olan tespitler için işaretleme yap
                    if is_true:
                        # True tespiti için özel path çizimi
                        points = np.array([
                            [x1, y1],  # Sol üst
                            [cx, y1 - height*0.1],  # Üst orta
                            [x2, y1],  # Sağ üst
                            [x2 + width*0.1, cy],  # Sağ orta
                            [x2, y2],  # Sağ alt
                            [cx, y2 + height*0.1],  # Alt orta
                            [x1, y2],  # Sol alt
                            [x1 - width*0.1, cy],  # Sol orta
                            [x1, y1]  # Başlangıç noktasına dön
                        ])
                        
                        # Path çizimi
                        fig.add_trace(go.Scatter(
                            x=points[:, 0],
                            y=points[:, 1],
                            mode='lines',
                            line=dict(
                                color=color,
                                width=2,
                            ),
                            name=base_label,  # Sadece bölge ismi
                            showlegend=True,
                            hoverinfo='text',
                            hovertext=f"{base_label}<br>Güven: {conf:.2%}"
                        ))
                        
                        # Etiket ekle
                        fig.add_annotation(
                            x=cx,
                            y=y1 - 10,
                            text=f"{base_label}<br>{conf:.2%}",  # Sadece bölge ismi
                            showarrow=False,
                            font=dict(
                                color='white',
                                size=12,
                                weight='bold'
                            ),
                            bgcolor=color,
                            opacity=0.8,
                            bordercolor=color,
                            borderwidth=2,
                            borderpad=4,
                            align='center'
                        )
                    else:
                        # False tespitler için sadece göstergede göster
                        fig.add_trace(go.Scatter(
                            x=[],
                            y=[],
                            mode='none',
                            name=f"{base_label} (Güven: {conf:.2%})",  # Sadece bölge ismi
                            showlegend=True
                        ))

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
                    pointpos=-1.8
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
                    meanline_visible=True
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
                    marker_color=['#1f77b4', '#ff7f0e']
                )
            ])
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
                opacity=0.7
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

    except Exception as e:
        error_msg = "An error occurred" if selected_language == "English" else "Hata oluştu"
        retry_msg = "Please try another image or refresh the page" if selected_language == "English" else "Lütfen farklı bir görüntü deneyin veya sayfayı yenileyin"
        st.error(f"{error_msg}: {str(e)}")
        st.error(retry_msg)
