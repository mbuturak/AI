# YOLOv11 X-Ray Analiz Uygulaması

Bu proje, YOLOv11 modeli kullanılarak X-Ray görüntülerinde nesne tespiti ve analizi yapan bir Streamlit uygulamasıdır.

## Özellikler

- X-Ray görüntülerinde otomatik nesne tespiti
- Detaylı bölgesel analiz
- Bölgesel yoğunluk haritası
- Boyut ve simetri analizi
- Güven skoru dağılımı
- Çoklu dil desteği (Türkçe/İngilizce)

## Kurulum

1. Gerekli bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. Sistem bağımlılıklarını yükleyin:
   ```bash
   sudo apt-get update
   sudo apt-get install -y $(cat packages.txt)
   ```

3. Model dosyasını `weights` klasörüne yerleştirin:
   - `weights/best.pt` konumuna YOLOv11 model dosyasını kopyalayın

## Kullanım

1. Uygulamayı başlatın:
   ```bash
   streamlit run app.py
   ```

2. Web tarayıcınızda açılan arayüzden:
   - Dil seçiminizi yapın
   - X-Ray görüntüsü yükleyin
   - Otomatik analiz sonuçlarını inceleyin

## Gereksinimler

Güncel gereksinimler `requirements.txt` dosyasında listelenmiştir:
- streamlit
- ultralytics
- opencv-python-headless
- pillow
- numpy
- plotly
- pandas
- scikit-image

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.
