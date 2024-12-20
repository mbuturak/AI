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
          - Yakın konumda birden fazla kemik yapısının bulunduğunu gösterir
          - Genellikle eklemler ve karmaşık anatomik yapılarda görülür
          - Detaylı inceleme gerektirebilecek alanları işaret edebilir
        
        - **Orta Yoğunluklu Bölgeler (Yeşil/Mavi)**:
          - Kemik yapılarının orta düzeyde yoğunlaştığı bölgeleri gösterir
          - Tipik olarak normal anatomik aralıkları temsil eder
        
        - **Düşük Yoğunluklu Bölgeler (Koyu Mavi)**:
          - Daha az yapının tespit edildiği alanları gösterir
          - Genellikle ana kemik grupları arasındaki boşlukları temsil eder
        
        Bu analiz şu konularda yardımcı olur:
        - Yapısal karmaşıklığın yüksek olduğu bölgeleri belirleme
        - Kemik dağılım modellerini anlama
        - Daha yakından inceleme gerektirebilecek bölgeleri vurgulama
        """,
        'size_title': "Boyut Analizi",
        'size_text': """
        **Boyut Analizi Yorumu:**
        
        Kutu grafiği, tespit edilen bölgelerin boyut dağılımını gösterir:
        
        - Her kutu farklı bir anatomik yapıyı temsil eder
        - Kutu, çeyrekler arası aralığı gösterir (25. ile 75. yüzdelik)
        - Kutunun içindeki çizgi medyan boyutu gösterir
        - Tekil noktalar spesifik tespitleri gösterir
        
        Bu analiz şunlarda yardımcı olur:
        - Farklı kemik yapılarının boyutlarını karşılaştırma
        - Olağandışı büyük veya küçük tespitleri belirleme
        - Her yapı için tipik boyut aralığını anlama
        """,
        'confidence_title': "Güven Skoru Analizi",
        'confidence_text': """
        **Güven Skoru Analizi:**
        
        Keman grafiği, modelin güven skorlarının dağılımını gösterir:
        
        - Geniş bölümler daha yaygın güven değerlerini gösterir
        - İçteki kutu çeyrekler ve medyanı gösterir
        - Yüksek skorlar (1.0'a yakın) güçlü güveni gösterir
        - Düşük skorlar daha az kesin tespitleri gösterir
        
        Bu analiz şunlarda yardımcı olur:
        - Tespitlerin güvenilirliğini değerlendirme
        - Model kesinliğindeki örüntüleri belirleme
        - Farklı yapılar için tespit kalitesini anlama
        """,
        'symmetry_title': "Simetri Analizi",
        'symmetry_text': """
        **Simetri Analizi:**
        
        Çubuk grafik, tespitlerin sol-sağ dağılımını karşılaştırır:
        
        - Her taraftaki tespit sayısını gösterir
        - Olası asimetrileri belirlemeye yardımcı olur
        - İki taraflı karşılaştırma için kullanışlıdır
        
        Şunlar için önemlidir:
        - Yapısal dengeyi kontrol etme
        - Potansiyel anomalileri belirleme
        - Karşılaştırmalı analizi destekleme
        """,
        'distance_title': "Mesafe Analizi",
        'distance_text': """
        **Mesafe Analizi:**
        
        Histogram, tespit edilen bölgeler arasındaki mesafelerin dağılımını gösterir:
        
        - X ekseni bölge çiftleri arasındaki mesafeyi gösterir
        - Y ekseni her mesafenin ne sıklıkta görüldüğünü gösterir
        - Tepeler yaygın aralık örüntülerini gösterir
        
        Bu analiz şunlarda yardımcı olur:
        - Uzamsal ilişkileri anlama
        - Tipik aralık örüntülerini belirleme
        - Olağandışı konumlandırmaları tespit etme
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
                width=None,  # Otomatik genişlik için None
                height=600,
                hoverlabel=dict(
                    namelength=-1
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
                    label = results[0].names[cls]
                    class_counts[label] = class_counts.get(label, 0) + 1

                # Pasta grafik
                pie_fig = go.Figure(data=[
                    go.Pie(
                        labels=list(class_counts.keys()),
                        values=list(class_counts.values()),
                        hole=.3,
                        textinfo='label+value+percent',  # Etiket, sayı ve yüzde göster
                        texttemplate='%{label}<br>%{value} adet<br>(%{percent})',  # Metin formatı
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
                hovertemplate="%{text}<br>Yoğunluk: %{z}<extra></extra>" if selected_language == "Türkçe" 
                            else "%{text}<br>Density: %{z}<extra></extra>",
                text=["Tespit Bölgesi" if selected_language == "Türkçe" else "Detection Region"] * len(centers_x)
            ))
            
            # X-ray görüntüsünü ekle
            density_fig.add_trace(go.Image(
                z=img_array, 
                opacity=0.5,
                hovertemplate="X-Ray<extra></extra>"
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
                size_labels.append(results[0].names[int(box.cls)])

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
                conf_labels.append(results[0].names[int(box.cls)])

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
                    pair_labels.append(f"{results[0].names[int(box1.cls)]} - {results[0].names[int(box2.cls)]}")

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
