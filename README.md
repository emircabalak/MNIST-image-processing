# MNIST El Yazısı Rakam Tanıma Projesi

> El yazısı ile yazılmış rakamları tanıyan makine öğrenimi projesi.  
> **Atakan** ve **Emir Cabalak** tarafından geliştirilmiştir.

## 🧠 Proje Hakkında

Bu proje, MNIST veri setini kullanarak 0–9 arasındaki el yazısı rakamları sınıflandırmak için makine öğrenimi modellerinden faydalanır. Amaç, veri ön işleme adımından model değerlendirmesine kadar verimli, anlaşılır ve yeniden üretilebilir bir yapay zeka pipeline’ı oluşturmaktır.

## ⚙️ Kullanılan Teknolojiler

- Python 3.x  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- TensorFlow / PyTorch (opsiyonel)  

## 🔍 Proje Adımları

1. **Veri Yükleme ve Görselleştirme**  
   MNIST veri seti incelendi ve temel istatistikler analiz edildi.

2. **Veri Ön İşleme**  
   - Normalizasyon
   - Görüntü yeniden boyutlandırma (varsa)
   - Eğitim/test ayrımı

3. **Model Geliştirme**  
   - Basit bir Lojistik Regresyon ve/veya CNN mimarisi kullanıldı.  
   - Model eğitildi ve doğruluk skoru üzerinden değerlendirildi.

4. **Sonuçların Değerlendirilmesi**  
   - Hata matrisleri  
   - Eğitim ve doğrulama başarı oranları  
